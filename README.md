# Sleep Mode

<p class="byline">Detecting sleep from the sensors a phone does not need permission for.<br>
Valmik Nahata · DSC 80, UC San Diego · <a href="http://extrasensory.ucsd.edu/">UCSD ExtraSensory dataset</a></p>

<div class="abstract" markdown="1">
Sleep tracking applications routinely request microphone and location access. Using 377,346 one-minute windows of smartphone telemetry from 60 users, I ask whether those permissions are necessary, or whether the sensors a phone reports without asking are already sufficient. Battery level turns out to carry a substantial and previously unremarked sleep signal, differing by 0.149 between sleeping and awake windows recorded in the same posture at the same hour of night. A decision tree using only motion and clock time reaches an F1 of 0.833 on fourteen held-out users, though a breakdown by hour shows it has learned a single rule that cannot represent sleep onset or wake time at all.
</div>

<div class="toc" markdown="0">
<p class="toc-title">Contents</p>
<a href="#introduction"><span class="num">1 · </span>Introduction</a>
<a href="#data-cleaning-and-exploratory-data-analysis"><span class="num">2 · </span>Data Cleaning and Exploratory Data Analysis</a>
<a href="#cleaning" class="sub">2.1 Cleaning</a>
<a href="#univariate-analysis" class="sub">2.2 Univariate analysis</a>
<a href="#bivariate-analysis" class="sub">2.3 Bivariate analysis</a>
<a href="#aggregation" class="sub">2.4 Aggregation</a>
<a href="#assessment-of-missingness"><span class="num">3 · </span>Assessment of Missingness</a>
<a href="#reasoning-about-the-mechanism" class="sub">3.1 Reasoning about the mechanism</a>
<a href="#dependency-tests" class="sub">3.2 Dependency tests</a>
<a href="#hypothesis-testing"><span class="num">4 · </span>Hypothesis Testing</a>
<a href="#framing-a-prediction-problem"><span class="num">5 · </span>Framing a Prediction Problem</a>
<a href="#baseline-model"><span class="num">6 · </span>Baseline Model</a>
<a href="#specification" class="sub">6.1 Specification</a>
<a href="#what-the-model-actually-learned" class="sub">6.2 What the model actually learned</a>
<a href="#final-model"><span class="num">7 · </span>Final Model</a>
<a href="#fairness-analysis"><span class="num">8 · </span>Fairness Analysis</a>
</div>

## Introduction

A phone can already tell when you are asleep. The open question is what it has to look at in order to do so.

Sleep staging applications ask for the microphone. Commercial trackers ask for location. Both sensors reveal far more than the state they are being used to infer, and neither permission, once granted, is restricted to nighttime. If the sensors a phone reports anyway are sufficient on their own, then a tracker that requests the invasive ones is making a product decision rather than satisfying a technical constraint. That distinction is testable, and this project tests it.

The question I investigate is how well a phone can identify sleep using only its least sensitive signals, meaning motion, battery, ringer, screen, and clock time, with the microphone and GPS excluded entirely.

The [UCSD ExtraSensory dataset](http://extrasensory.ucsd.edu/) is unusually well suited to the comparison. It contains a year of in-the-wild smartphone and smartwatch telemetry from 60 volunteers, collected on this campus between June 2015 and June 2016, in which every row is a single one-minute window described by roughly 225 pre-computed sensor features alongside 51 context labels the participants reported about themselves. Because it carries both the invasive sensors and the innocuous ones on identical windows, the two can be compared directly rather than across separate studies.

I use all 377,346 windows from all 60 users. The source files carry 278 columns; I retain 30, covering the sensor families that could plausibly bear on sleep, the labels required to define and frame the problem, and two sensors that exist in the data purely so that I can demonstrate excluding them.

| Column | Description |
| --- | --- |
| `uuid` | Anonymized user identifier, recovered from the filename |
| `timestamp` | Unix time at the start of the one-minute window |
| `raw_acc:magnitude_stats:std` | Standard deviation of accelerometer magnitude, the primary stillness measure |
| `raw_acc:magnitude_stats:mean` | Mean accelerometer magnitude, near 1 g when the phone is at rest |
| `raw_acc:magnitude_stats:value_entropy` | Entropy of the accelerometer magnitude distribution |
| `proc_gyro:magnitude_stats:std` | Rotation variability, which captures handling that translation misses |
| `lf_measurements:battery_level` | Battery charge as a fraction from 0 to 1 |
| `discrete:battery_state:*` | One-hot family covering unplugged, discharging, not charging, charging, full |
| `discrete:ringer_mode:*` | One-hot family covering normal, silent with vibrate, silent without vibrate |
| `discrete:app_state:*` | One-hot family recording whether the app was foreground, background, or inactive |
| `lf_measurements:screen_brightness` | Screen brightness from 0 to 1 |
| `lf_measurements:light` | Ambient light, already log-scaled at source |
| `discrete:wifi_status:is_reachable_via_wifi` | Whether the phone held a WiFi connection |
| `discrete:on_the_phone:is_True` | Whether a call was active |
| `label:SLEEPING` | Self-reported sleep, taking values 1, 0, or missing, and serving as the response variable |
| `label:LYING_DOWN` | Self-reported posture, used to separate sleep from awake rest |
| `label:LOC_home` | Self-reported location |
| `location:log_diameter` | Log spatial spread of GPS fixes, withheld from the model deliberately |
| `audio_properties:max_abs_value` | Log peak microphone amplitude, also withheld deliberately |
| `raw_magnet:magnitude_stats:std` | Magnetometer variability, used as a control in the missingness tests |

## Data Cleaning and Exploratory Data Analysis

### Cleaning

Each user occupies a separate compressed file named for their anonymized identifier, and that identifier appears nowhere in the file contents. I recovered it from the filename and attached it as a column before concatenating, since every subsequent step depends on knowing which user a window came from, and the train-test split is drawn along user boundaries.

Timestamps arrive as seconds since epoch, which is unusable for a question about sleep. The study ran at UC San Diego, so I converted to `America/Los_Angeles`. The choice is not cosmetic. A conversion that ignores daylight saving displaces half the year by an hour, and an hour is a large error when the entire question concerns where sleep falls on the clock.

The `discrete:` sensors arrive pre-encoded as one-hot families, so `battery_state` is spread across six binary columns. That shape is wrong for both grouping and for `OneHotEncoder`, and it conceals the fact that the categories are mutually exclusive, so I collapsed each family back into a single nominal column. Every family also carries a `:missing` indicator, which I mapped to a null rather than treating as a category. For `ringer_mode` that indicator is set on 58.6% of windows, almost all originating from iPhones, because iOS does not expose ringer state to third-party applications. Encoding a device limitation as though it meant the phone was silent would fabricate data.

I dropped `lf_measurements:proximity`, which takes the value 0 in all 377,346 windows and therefore cannot carry information.

The sleep label is absent in 24.4% of windows, and I left it absent. Filling it with 0 would assert that every unlabeled window is a waking one, which is close to the reverse of the truth, and the following section exists to examine why those windows are missing in the first place.

Finally, a night crosses midnight, so grouping by calendar date bisects every sleep episode. I shifted each timestamp back twelve hours before taking the date, which defines a night as a noon-to-noon block named for the evening on which it begins, and lets a derived `hours_since_noon` run from 0 to 24 across that block with no wraparound.

Two columns look as though they need a log transform and do not receive one. Both `lf_measurements:light` and `audio_properties:max_abs_value` are already log-scaled in the source data.

| uuid | datetime | hour | acc_std | battery_level | battery_state | ringer_mode | asleep |
|:---|:---|---:|---:|---:|:---|---:|---:|
| 00EABED2-271D-49D8-B599-1D4A09240601 | 2015-10-05 14:06:01-07:00 | 14.1 | 0.003529 | 0.46 | charging | nan | 0 |
| 00EABED2-271D-49D8-B599-1D4A09240601 | 2015-10-05 14:07:01-07:00 | 14.1167 | 0.004172 | 0.46 | charging | nan | 0 |
| 00EABED2-271D-49D8-B599-1D4A09240601 | 2015-10-05 14:08:01-07:00 | 14.1333 | 0.003667 | 0.46 | charging | nan | 0 |
| 00EABED2-271D-49D8-B599-1D4A09240601 | 2015-10-05 14:09:01-07:00 | 14.15 | 0.003541 | 0.46 | charging | nan | 0 |
| 00EABED2-271D-49D8-B599-1D4A09240601 | 2015-10-05 14:10:31-07:00 | 14.1667 | 0.037653 | 0.47 | unplugged | nan | 0 |

### Univariate analysis

<div class="figure" markdown="0">
<iframe src="assets/motion-distribution.html" frameborder="0"></iframe>
<p class="caption"><span class="lab">Figure 1.</span> Distribution of accelerometer magnitude standard deviation across all 377,346 windows, on a log scale.</p>
</div>

Accelerometer variability is bimodal. A tall, narrow mode near 10<sup>-2.8</sup> corresponds to a phone lying completely still, and a broad, low mode near 10<sup>-1.1</sup> corresponds to one being carried or handled, with a clear trough separating them. A single threshold on this column therefore partitions most windows cleanly. The difficulty is that the still mode contains both a sleeping user and a phone abandoned on a desk, so stillness alone cannot resolve the question.

<div class="figure" markdown="0">
<iframe src="assets/sleep-duration.html" frameborder="0"></iframe>
<p class="caption"><span class="lab">Figure 2.</span> Recorded sleep per night across the 194 nights carrying at least 400 labeled windows and two hours of sleep.</p>
</div>

Nightly sleep centers on a median of 7.0 hours with a long left tail. Some of that tail reflects genuinely short nights and some reflects interrupted recording, since a phone that stops sampling at three in the morning produces a night indistinguishable from an early waking.

### Bivariate analysis

<div class="figure" markdown="0">
<iframe src="assets/sleep-clock.html" frameborder="0"></iframe>
<p class="caption"><span class="lab">Figure 3.</span> Proportion of labeled windows reported as sleeping, by local hour of day.</p>
</div>

Sleep peaks near 90% between three and five in the morning and falls below 2% in the late afternoon, with steep transitions around eleven at night and eight in the morning. The daytime floor does not reach zero, and that matters, because naps and irregular schedules mean clock time can never fully determine the label on its own.

The harder comparison is between sleep and waking rest, since lying in bed at one in the morning resembles sleeping at one in the morning. Restricting to windows in which the user reported lying down between ten at night and six in the morning holds posture and clock time approximately fixed, which isolates what remains.

<div class="figure" markdown="0">
<iframe src="assets/battery-by-state.html" frameborder="0"></iframe>
<p class="caption"><span class="lab">Figure 4.</span> Battery level for sleeping and waking windows recorded while lying down between 22:00 and 06:00, across 67,328 windows from 52 users.</p>
</div>

The separation falls almost entirely in one band. Full batteries account for 58% of sleeping windows and 29% of waking ones, while waking windows concentrate in the middle range where a phone has been in use through the evening. The plausible mechanism is the bedtime routine, in which the phone goes on the charger, fills, and remains there, whereas someone awake at two in the morning is draining it.

### Aggregation

Sleep rate broken down simultaneously by clock time and battery state, over all 285,268 labeled windows, separates the contribution of each.

| Time of day | Unplugged | Discharging | Not charging | Charging | Full |
|:---|---:|---:|---:|---:|---:|
| 00-04 | 0.647 | 0.558 | 0.967 | 0.845 | 0.904 |
| 04-08 | 0.471 | 0.564 | 0.708 | 0.791 | 0.907 |
| 08-12 | 0.063 | 0.083 | 0.297 | 0.082 | 0.657 |
| 12-16 | 0.042 | 0.024 | 0.000 | 0.030 | 0.089 |
| 16-20 | 0.039 | 0.007 | 0.000 | 0.028 | 0.219 |
| 20-24 | 0.094 | 0.033 | 0.527 | 0.327 | 0.293 |

A full battery exceeds an unplugged one in every time bucket, frequently by a wide margin. Between eight in the morning and noon the sleep rate moves from 0.06 unplugged to 0.66 full, which is the difference between almost certainly awake and probably still in bed. Battery state therefore carries information that clock time does not, which is the premise the model rests on.

Active charging behaves far less consistently and in the afternoon falls slightly below unplugged. What tracks sleep is accumulated charge rather than the act of drawing power, since a phone plugged in at two in the afternoon is a phone at a desk, while a phone at full charge at six in the morning has spent eight hours on a nightstand.

The `not_charging` column rests on 3,986 labeled windows spread over six buckets, and its 00:00 to 04:00 cell on 30 of them, so those cells swing widely and should not be read closely.

## Assessment of Missingness

### Reasoning about the mechanism

I believe `label:SLEEPING` is MNAR.

Labels are self-reported through the ExtraSensory application, and a sleeping person cannot report anything. Every label covering a period of sleep is either entered beforehand or reconstructed after waking, so the state of being asleep is itself what prevents the label from being recorded. The probability that a value is missing therefore depends on the value it would have taken, which is what MNAR describes. The 24.4% missing rate is also not spread evenly, but concentrated in long overnight blocks in which a user went to bed without setting a label.

The additional data that would render this MAR is the application's own interaction log, recording when the user was prompted, when the app was brought to the foreground, and when a notification was dismissed. Knowing that a window fell within a stretch in which the user was never prompted would explain the missingness through the prompt schedule rather than through the sleep state, making it MAR conditional on prompt history. The public release does not include that log.

### Dependency tests

One methodological point governs everything below. The rows are one-minute windows drawn from continuous recording, so consecutive rows are nearly identical, and treating all 377,346 as independent draws drives every permutation test to a p-value of zero, including tests on columns with no plausible relationship to the outcome. To recover an effective sample size closer to the true one, I draw a single window per user per hour, which leaves 7,376 rows, still substantial but no longer dominated by within-session autocorrelation.

For the first test the null hypothesis is that the distribution of battery level is the same whether or not the sleep label is missing, against the alternative that the two distributions differ. The test statistic is the absolute difference in mean battery level between the missing and non-missing groups, evaluated at a significance level of 0.05. Mean battery level is 0.605 where the label is absent and 0.672 where it is present, giving an observed statistic of 0.067.

<div class="figure" markdown="0">
<iframe src="assets/missingness-battery.html" frameborder="0"></iframe>
<p class="caption"><span class="lab">Figure 5.</span> Empirical null distribution over 5,000 permutations, with the observed statistic marked.</p>
</div>

The observed gap does not occur in any of 5,000 permutations, so the p-value falls below 0.001 and I reject the null. Windows carrying a missing sleep label sit at a lower battery level. The mechanism is direct, in that a phone running low belongs to someone who has stopped engaging with the application, and a phone that dies overnight stops recording labels entirely.

The second test uses magnetometer variability, which measures fluctuation in the local magnetic field and responds to nearby metal and electronics. No route connects that to whether a participant remembered to label their sleep, which makes it a suitable control. The null hypothesis is that the distribution of magnetometer variability is the same whether or not the sleep label is missing, against the alternative that the two differ, using the absolute difference in means as the statistic at the same 0.05 level. The observed statistic is 0.392 with a p-value of 0.284, well above the threshold, so I fail to reject the null.

| Column | Observed statistic | p-value | n |
| --- | ---: | ---: | ---: |
| `battery_level` | 0.0669 | < 0.001 | 7,369 |
| `magnet_std` | 0.3916 | 0.284 | 6,719 |

The sleep label therefore goes missing more often when the battery is low and at an unchanged rate regardless of the magnetic environment, which is the pattern expected if missingness is governed by whether the user and the phone were in any condition to record something. It is also the reason every subsequent section uses only labeled windows, and that filter biases the sample toward better-charged phones. This limits everything that follows and is not a footnote.

## Hypothesis Testing

The exploratory work suggests that among people lying in bed at night, those asleep have fuller batteries. The claim deserves a formal test, because battery level is the cheapest sleep signal available. It requires no permission on any platform, discloses nothing about location or speech, and is already visible on screen.

Restricting to windows in which the user reported lying down between ten at night and six in the morning holds posture and clock time approximately fixed, so the comparison runs between sleep and waking rest rather than between night and day. I use the same one-window-per-user-hour subsample described above, for the same reason, which leaves 1,274 windows from 52 users.

The null hypothesis is that among those windows the mean battery level is the same for windows labeled sleeping and windows labeled not sleeping, with any observed difference attributable to chance. The alternative is that the mean battery level is higher when the user reported sleeping. The test statistic is the difference in mean battery level, sleeping minus waking, signed rather than absolute because the alternative is directional, and a difference in means rather than a total variation or Kolmogorov-Smirnov statistic because battery level is quantitative and the claim concerns level rather than shape. I evaluate at a significance level of 0.05 using a permutation test with 10,000 repetitions.

<div class="figure" markdown="0">
<iframe src="assets/hypothesis-null.html" frameborder="0"></iframe>
<p class="caption"><span class="lab">Figure 6.</span> Empirical null distribution over 10,000 permutations, with the observed difference of 0.149 marked.</p>
</div>

The observed difference is 0.149 with a p-value below 0.001. Sleeping windows sit roughly fifteen percentage points higher in charge than waking ones at the same hour and in the same posture, and across 10,000 permutations no rearrangement of the labels produced a difference that large, so I reject the null at the 0.05 level.

This constitutes evidence against the null rather than proof of a mechanism. A permutation test on observational data cannot exclude a common cause, and the most likely candidate is the bedtime routine itself, since plugging in and going to sleep are the same act for most people, which would make battery level a proxy for a habit rather than for sleep. That is acceptable for prediction and is precisely why the model relies on it, but it would not support a causal claim.

Two limitations bound the estimate. The waking group contains only 142 windows after subsampling, because people who report lying down at night are usually asleep, so the waking mean is the noisier half of the comparison. The test also runs only on windows where the label exists, which the preceding section showed skews toward better-charged phones. Both considerations argue for treating 0.149 as an effect size with wide error bars rather than a point estimate.

One further observation is that instantaneous charging state proved a substantially weaker signal than accumulated charge. Whether a phone is drawing power at this moment says less than whether it has been connected long enough to fill.

## Framing a Prediction Problem

Given one minute of passive phone telemetry, I predict whether the user would label that window as sleeping, which is a binary classification task. The response variable is `label:SLEEPING` over the 285,254 windows in which it is present and the accelerometer recorded. It is the direct target of everything above, since each preceding section examined what distinguishes sleep from its absence, and this column encodes that distinction.

I evaluate with F1 on the sleeping class. Accuracy is unsuitable because only 29.1% of labeled windows are sleep, so a model that always predicts waking scores 70.9% while being useless. The two error types also carry different costs and both matter, in that a false negative fragments a real night and understates its duration, while a false positive counts an hour of lying in the dark as sleep and overstates it. F1 requires precision and recall to be acceptable simultaneously, which is what a usable sleep summary demands. I report precision, recall, and accuracy alongside it so that the tradeoff remains visible.

Every feature is a passive sensor reading drawn from the same one-minute window and available on the device the moment that window closes, so nothing looks forward and no label enters as an input. I exclude all GPS and microphone features despite their presence in the data and their predictive strength, because the premise of the project is what the remaining sensors can achieve alone. I also exclude every other `label:` column, including `LYING_DOWN`, which was useful for framing the hypothesis test but is itself self-reported and would not exist at prediction time.

The split runs along user boundaries rather than rows. A random row split would place minutes from the same night on both sides, allowing a model to score well by memorizing individual habits, and the object here is a tracker that works on a phone it has never encountered. I therefore hold out 25% of users in full, leaving 39 training users covering 208,526 windows and 14 test users covering 76,728, with no data shared between them.

## Baseline Model

### Specification

The baseline uses the two features the exploratory analysis pointed to most directly and nothing else.

| Feature | Type | Encoding |
| --- | --- | --- |
| `acc_std` | Quantitative | Passed through unchanged, since a decision tree splits on thresholds and monotone rescaling would alter nothing |
| `tod` | Nominal | One-hot encoded into six four-hour buckets with unknown categories ignored |

The second is a derived column rather than a raw one. Clock hour is cyclic and has no meaningful zero, so supplying it as a number would let the model treat 23:00 and 00:00 as maximally distant. Bucketing into six blocks and one-hot encoding removes the ordering entirely. The encoding is blunt, and improving it is the first item in the plan below.

The estimator is a depth-3 decision tree, kept shallow deliberately, because with two inputs and clear threshold structure in both, additional depth would fit user-specific quirks that do not transfer to held-out participants. Feature transformation and model fitting occupy a single scikit-learn pipeline.

| Split | F1 | Precision | Recall | Accuracy |
|:---|---:|---:|---:|---:|
| Train | 0.790 | 0.799 | 0.781 | 0.884 |
| Test, unseen users | 0.833 | 0.877 | 0.794 | 0.897 |

### What the model actually learned

The result is partly good. An F1 of 0.833 on fourteen users the model has never seen is genuine, and it stands well above the 0.488 obtained by predicting sleep on every window. Two features and three splits recover most of the available signal, which indicates the problem is easy in the bulk.

Breaking recall down by hour shows that the headline score is misleading. Recall exceeds 0.94 at every hour from midnight through eight in the morning and is exactly zero at every other hour. A depth-3 tree given these two inputs has discovered a single rule, that sleeping means still and between midnight and eight, and consequently misses every sleeping minute reported at eleven at night or after eight in the morning, amounting to 4,491 windows in the test set alone.

Three consequences follow. The model cannot represent sleep onset or wake time, which are the two quantities a sleep tracker exists to report and which fall precisely in the hours where recall vanishes; an overall recall of 0.794 conceals this because the midnight-to-eight band contains most of the sleep. Test F1 also exceeds train F1, at 0.833 against 0.790, which does not indicate strong generalization but reflects the held-out users sleeping more, 32.3% against 27.9%, inflating F1 on the positive class and marking a single fourteen-user split as a noisy estimate. Finally the two features compensate for each other badly, since stillness cannot separate sleep from a phone on a desk, clock time cannot separate sleep from lying awake at two in the morning, and neither feature knows anything about the preceding twenty minutes, when sleep is sustained stillness rather than any single still minute.

## Final Model

In progress. Four additions are planned, each aimed at the hours where recall currently vanishes.

Rolling stillness, computed as the mean of `acc_std` over a centered 31-minute window within each user, addresses the largest structural gap, since the baseline treats every minute as independent when a single still minute is ambiguous and half an hour of stillness is not. Time since the phone was last handled, measured in minutes since the most recent foreground or call window, encodes the transition into sleep rather than the state, which is what the failures at eleven at night require. Cyclic hour, replacing the six buckets with the sine and cosine of the clock angle, places 23:50 adjacent to 00:10 and removes the hard boundary that produced the failure in the first place. The privacy-safe phone state, covering `battery_level`, `battery_state`, `ringer_mode`, and `screen_brightness`, supplies information that the hypothesis test established is present exactly where clock time and stillness both fail.

The estimator moves to a random forest, tuning maximum depth and minimum samples per leaf by grid search on the same split and the same metric, so that the comparison against the baseline is direct.

## Fairness Analysis

In progress. The planned comparison evaluates model F1 on users whose sleep is concentrated overnight against users with irregular or daytime schedules, tested by permutation. A model leaning as heavily on clock time as this one does should be expected to underperform on the second group, and the purpose of the analysis is to measure the size of that gap.
