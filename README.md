# Sleep Mode

<p class="byline">Detecting sleep from the sensors a phone does not need permission for.<br>
Valmik Nahata · DSC 80, UC San Diego · <a href="http://extrasensory.ucsd.edu/">UCSD ExtraSensory dataset</a></p>

<div class="abstract" markdown="1">
Sleep tracking applications routinely request microphone and location access. Using 377,346 one-minute windows of smartphone telemetry from 60 users, I ask whether those permissions are necessary, or whether the sensors a phone reports without asking are already sufficient. Battery level turns out to carry a substantial and previously unremarked sleep signal, differing by 0.149 between sleeping and awake windows recorded in the same posture at the same hour of night. A decision tree using only motion and clock time reaches an F1 of 0.833 on fourteen held out users, though a breakdown by hour shows it has learned a single rule that cannot represent sleep onset or wake time at all. Adding trailing stillness, time since the phone was last handled, a cyclic clock, and the phone's own power and ringer state lifts a random forest to 0.870 and, more importantly, recovers the hours where sleep actually begins and ends. The same model is roughly nine points of F1 worse for people whose schedules are irregular.
</div>

<div class="toc" markdown="0">
<p class="toc-title">Contents</p>
<a href="#introduction"><span class="num">1 · </span>Introduction</a>
<a href="#data-cleaning-and-exploratory-data-analysis"><span class="num">2 · </span>Data Cleaning and Exploratory Data Analysis</a>
<a href="#cleaning" class="sub">2.1 Cleaning</a>
<a href="#what-one-night-looks-like" class="sub">2.2 What one night looks like</a>
<a href="#univariate-analysis" class="sub">2.3 Univariate analysis</a>
<a href="#bivariate-analysis" class="sub">2.4 Bivariate analysis</a>
<a href="#aggregation" class="sub">2.5 Aggregation</a>
<a href="#assessment-of-missingness"><span class="num">3 · </span>Assessment of Missingness</a>
<a href="#reasoning-about-the-mechanism" class="sub">3.1 Reasoning about the mechanism</a>
<a href="#dependency-tests" class="sub">3.2 Dependency tests</a>
<a href="#hypothesis-testing"><span class="num">4 · </span>Hypothesis Testing</a>
<a href="#framing-a-prediction-problem"><span class="num">5 · </span>Framing a Prediction Problem</a>
<a href="#baseline-model"><span class="num">6 · </span>Baseline Model</a>
<a href="#specification" class="sub">6.1 Specification</a>
<a href="#what-the-model-actually-learned" class="sub">6.2 What the model actually learned</a>
<a href="#final-model"><span class="num">7 · </span>Final Model</a>
<a href="#engineered-features" class="sub">7.1 Engineered features</a>
<a href="#model-selection" class="sub">7.2 Model selection</a>
<a href="#result" class="sub">7.3 Result</a>
<a href="#fairness-analysis"><span class="num">8 · </span>Fairness Analysis</a>
<a href="#links"><span class="num">9 · </span>Links</a>
</div>

## Introduction

A phone can already tell when you are asleep. The open question is what it has to look at in order to do so.

Sleep staging applications ask for the microphone. Commercial trackers ask for location. Both sensors reveal far more than the state they are being used to infer, and neither permission, once granted, is restricted to nighttime. If the sensors a phone reports anyway are sufficient on their own, then a tracker that requests the invasive ones is making a product decision, not satisfying a technical constraint. That distinction is testable, and this project tests it.

The question I investigate is how well a phone can identify sleep using only its least sensitive signals, meaning motion, battery, ringer, screen, and clock time, with the microphone and GPS excluded entirely.

The [UCSD ExtraSensory dataset](http://extrasensory.ucsd.edu/) is unusually well suited to the comparison. It contains a year of real world smartphone and smartwatch telemetry from 60 volunteers, collected on this campus between June 2015 and June 2016, in which every row is a single one-minute window described by roughly 225 pre-computed sensor features alongside 51 context labels the participants reported about themselves. Because it carries both the invasive sensors and the innocuous ones on identical windows, the two can be compared directly instead of across separate studies.

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

The `discrete:` sensors arrive already encoded as one-hot families, so `battery_state` is spread across six binary columns. That shape is wrong for both grouping and for `OneHotEncoder`, and it conceals the fact that the categories are mutually exclusive, so I collapsed each family back into a single nominal column. Every family also carries a `:missing` indicator, which I mapped to a null instead of treating it as a category. For `ringer_mode` that indicator is set on 58.6% of windows, almost all originating from iPhones, because iOS does not expose ringer state to outside applications. Encoding a device limitation as though it meant the phone was silent would fabricate data.

I dropped `lf_measurements:proximity`, which takes the value 0 in all 377,346 windows and therefore cannot carry information.

The sleep label is absent in 24.4% of windows, and I left it absent. Filling it with 0 would assert that every unlabeled window is a waking one, which is close to the reverse of the truth, and the following section exists to examine why those windows are missing in the first place.

Finally, a night crosses midnight, so grouping by calendar date bisects every sleep episode. I shifted each timestamp back twelve hours before taking the date, which defines a night as a block running noon to noon named for the evening on which it begins, and lets a derived `hours_since_noon` run from 0 to 24 across that block with no wraparound.

Two columns look as though they need a log transform and do not receive one. Both `lf_measurements:light` and `audio_properties:max_abs_value` are already log-scaled in the source data.

| uuid | datetime | hour | acc_std | battery_level | battery_state | ringer_mode | asleep |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 00EABED2-271D-49D8-B599-1D4A09240601 | 2015-10-05 14:06:01-07:00 | 14.1 | 0.003529 | 0.46 | charging | nan | 0 |
| 00EABED2-271D-49D8-B599-1D4A09240601 | 2015-10-05 14:07:01-07:00 | 14.1167 | 0.004172 | 0.46 | charging | nan | 0 |
| 00EABED2-271D-49D8-B599-1D4A09240601 | 2015-10-05 14:08:01-07:00 | 14.1333 | 0.003667 | 0.46 | charging | nan | 0 |
| 00EABED2-271D-49D8-B599-1D4A09240601 | 2015-10-05 14:09:01-07:00 | 14.15 | 0.003541 | 0.46 | charging | nan | 0 |
| 00EABED2-271D-49D8-B599-1D4A09240601 | 2015-10-05 14:10:31-07:00 | 14.1667 | 0.037653 | 0.47 | unplugged | nan | 0 |

### What one night looks like

Before any aggregation, here is a single continuous day of recording from one participant. The night of 11 October 2015 gave 1,443 windows, 99.3% of them labeled, with no missing accelerometer or battery readings anywhere in the block.

<div class="figure" markdown="0">
<iframe src="assets/one-night.html" frameborder="0"></iframe>
<p class="caption"><span class="lab">Figure 1.</span> Motion and battery for one participant across 24 hours. Shaded blocks are windows the participant labeled as sleeping.</p>
</div>

The shaded block is what the participant reported. Motion collapses to the sensor floor and stays there for hours, then the battery climbs from a third of a charge to full and holds. Both signals move at the same moment, and neither one needed a microphone or a location fix to say so. Everything after this section is an attempt to detect that pattern without the shading.

### Univariate analysis

<div class="figure" markdown="0">
<iframe src="assets/motion-distribution.html" frameborder="0"></iframe>
<p class="caption"><span class="lab">Figure 2.</span> Distribution of accelerometer magnitude standard deviation across all 377,346 windows, on a log scale.</p>
</div>

Accelerometer variability is bimodal. A tall, narrow mode near 10<sup>-2.8</sup> corresponds to a phone lying completely still, and a broad, low mode near 10<sup>-1.1</sup> corresponds to one being carried or handled, with a clear trough separating them. A single threshold on this column therefore partitions most windows cleanly. The difficulty is that the still mode contains both a sleeping user and a phone abandoned on a desk, so stillness alone cannot resolve the question.

<div class="figure" markdown="0">
<iframe src="assets/sleep-duration.html" frameborder="0"></iframe>
<p class="caption"><span class="lab">Figure 3.</span> Recorded sleep per night across the 194 nights carrying at least 400 labeled windows and two hours of sleep.</p>
</div>

Nightly sleep centers on a median of 7.0 hours with a long left tail. Some of that tail reflects genuinely short nights and some reflects interrupted recording, since a phone that stops sampling at three in the morning produces a night indistinguishable from an early waking.

### Bivariate analysis

<div class="figure" markdown="0">
<iframe src="assets/sleep-clock.html" frameborder="0"></iframe>
<p class="caption"><span class="lab">Figure 4.</span> Proportion of labeled windows reported as sleeping, by local hour of day.</p>
</div>

Sleep peaks near 90% between three and five in the morning and falls below 2% in the late afternoon, with steep transitions around eleven at night and eight in the morning. The daytime floor does not reach zero, and that matters, because naps and irregular schedules mean clock time can never fully determine the label on its own.

The harder comparison is between sleep and waking rest, since lying in bed at one in the morning resembles sleeping at one in the morning. Restricting to windows in which the user reported lying down between ten at night and six in the morning holds posture and clock time approximately fixed, which isolates what remains.

<div class="figure" markdown="0">
<iframe src="assets/battery-by-state.html" frameborder="0"></iframe>
<p class="caption"><span class="lab">Figure 5.</span> Battery level for sleeping and waking windows recorded while lying down between 22:00 and 06:00, across 67,328 windows from 52 users.</p>
</div>

The separation falls almost entirely in one band. Full batteries account for 58% of sleeping windows and 29% of waking ones, while waking windows concentrate in the middle range where a phone has been in use through the evening. The plausible mechanism is the bedtime routine, in which the phone goes on the charger, fills, and remains there, whereas someone awake at two in the morning is draining it.

### Aggregation

Sleep rate broken down simultaneously by clock time and battery state, over all 285,268 labeled windows, separates the contribution of each.

| Time of day | Unplugged | Discharging | Not charging | Charging | Full |
| --- | ---: | ---: | ---: | ---: | ---: |
| 00-04 | 0.647 | 0.558 | 0.967 | 0.845 | 0.904 |
| 04-08 | 0.471 | 0.564 | 0.708 | 0.791 | 0.907 |
| 08-12 | 0.063 | 0.083 | 0.297 | 0.082 | 0.657 |
| 12-16 | 0.042 | 0.024 | 0.000 | 0.030 | 0.089 |
| 16-20 | 0.039 | 0.007 | 0.000 | 0.028 | 0.219 |
| 20-24 | 0.094 | 0.033 | 0.527 | 0.327 | 0.293 |

<div class="figure" markdown="0">
<iframe src="assets/sleep-heatmap.html" frameborder="0"></iframe>
<p class="caption"><span class="lab">Figure 6.</span> The same table as a heatmap. Darker cells carry a higher share of sleeping windows.</p>
</div>

A full battery exceeds an unplugged one in every time bucket, frequently by a wide margin. Between eight in the morning and noon the sleep rate moves from 0.06 unplugged to 0.66 full, which is the difference between almost certainly awake and probably still in bed. Battery state therefore carries information that clock time does not, which is the premise the model rests on.

Active charging behaves far less consistently and in the afternoon falls slightly below unplugged. What tracks sleep is accumulated charge, not the act of drawing power, since a phone plugged in at two in the afternoon is a phone at a desk, while a phone at full charge at six in the morning has spent eight hours on a nightstand.

The `not_charging` column rests on 3,986 labeled windows spread over six buckets, and its 00:00 to 04:00 cell on 30 of them, so those cells swing widely and should not be read closely.

## Assessment of Missingness

### Reasoning about the mechanism

I believe `label:SLEEPING` is MNAR.

Labels are self-reported through the ExtraSensory application, and a sleeping person cannot type. Every label covering a period of sleep is therefore either entered beforehand or reconstructed after waking, which means it is assigned to a remembered block of time, not to the minute it describes. Whether a window carries a label depends on whether it fell inside an activity the participant could later recall as one coherent stretch, and that depends on what the activity was. The probability a value is missing depends on the value it would have taken, which is what MNAR describes.

Where the labels actually go missing is worth looking at directly, because the obvious guess is wrong.

<div class="figure" markdown="0">
<iframe src="assets/missing-by-hour.html" frameborder="0"></iframe>
<p class="caption"><span class="lab">Figure 7.</span> Share of windows reported as sleeping and share with no sleep label, by local hour.</p>
</div>

Missing labels are at their least common in the middle of the night and their most common in the evening. Across the 24 hours the correlation between the two curves is negative at 0.76. Between three and five in the morning about 18% of windows are unlabeled, while between ten and midnight the figure reaches 31%.

This is the retrospective mechanism showing itself. Participants wake up and record the whole night as one block, so the hours they were unconscious end up better documented than the hours they were awake and busy. Evening minutes are fragmented across cooking, commuting, and television, none of which gets recalled cleanly the next morning. The label is missing when the underlying activity was forgettable, and being asleep turns out to be one of the most memorable states there is.

The additional data that would make this MAR is the application’s own interaction log, recording when the user was prompted, when the app was opened, and when a notification went unanswered. Knowing that a window fell in a stretch where nobody was ever prompted would explain the missingness through the prompt schedule instead of through the activity, making it MAR conditional on prompt history. The public release does not include that log.

### Dependency tests

One methodological point governs everything below. The rows are one-minute windows drawn from continuous recording, so consecutive rows are nearly identical, and treating all 377,346 as independent draws drives every permutation test to a p-value of zero, including tests on columns with no plausible relationship to the outcome. To recover an effective sample size closer to the true one, I draw a single window per user per hour, which leaves 7,376 rows, still substantial but no longer dominated by within-session autocorrelation.

For the first test the null hypothesis is that the distribution of battery level is the same whether or not the sleep label is missing, against the alternative that the two distributions differ. The test statistic is the absolute difference in mean battery level between the missing and non-missing groups, evaluated at a significance level of 0.05. Mean battery level is 0.605 where the label is absent and 0.672 where it is present, giving an observed statistic of 0.067.

<div class="figure" markdown="0">
<iframe src="assets/missingness-battery.html" frameborder="0"></iframe>
<p class="caption"><span class="lab">Figure 8.</span> Empirical null distribution over 5,000 permutations, with the observed statistic marked.</p>
</div>

The observed difference does not occur in any of 5,000 permutations, so the p-value falls below 0.001 and I reject the null. Windows carrying a missing sleep label sit at a lower battery level. The mechanism is direct, in that a phone running low belongs to someone who has stopped engaging with the application, and a phone that dies overnight stops recording labels entirely.

The second test uses magnetometer variability, which measures fluctuation in the local magnetic field and responds to nearby metal and electronics. No route connects that to whether a participant remembered to label their sleep, which makes it a suitable control. The null hypothesis is that the distribution of magnetometer variability is the same whether or not the sleep label is missing, against the alternative that the two differ, using the absolute difference in means as the statistic at the same 0.05 level. The observed statistic is 0.392 with a p-value of 0.284, well above the threshold, so I fail to reject the null.

| Column | Observed statistic | p-value | n |
| --- | ---: | --- | ---: |
| `battery_level` | 0.0669 | < 0.001 | 7,369 |
| `magnet_std` | 0.3916 | 0.284 | 6,719 |

The sleep label therefore goes missing more often when the battery is low and at an unchanged rate regardless of the magnetic environment, which is the pattern expected if missingness is governed by whether the user and the phone were in any condition to record something. It is also the reason every subsequent section uses only labeled windows, and that filter biases the sample toward better charged phones. This limits everything that follows and is not a footnote.

## Hypothesis Testing

The exploratory work suggests that among people lying in bed at night, those asleep have fuller batteries. The claim deserves a formal test, because battery level is the cheapest sleep signal available. It requires no permission on any platform, discloses nothing about location or speech, and is already visible on screen.

Restricting to windows in which the user reported lying down between ten at night and six in the morning holds posture and clock time approximately fixed, so the comparison runs between sleep and waking rest, not between night and day. I use the same hourly subsample approach described above, which leaves 1,274 windows from 52 users.

The null hypothesis is that among those windows the mean battery level is the same for windows labeled sleeping and windows labeled not sleeping, with any observed difference attributable to chance. The alternative is that the mean battery level is higher when the user reported sleeping. The test statistic is the difference in mean battery level, sleeping minus waking, signed, because the alternative is directional, and a difference in means instead of a total variation or Kolmogorov-Smirnov statistic because battery level is quantitative and the claim concerns level, not shape. I evaluate at a significance level of 0.05 using a permutation test with 10,000 repetitions.

<div class="figure" markdown="0">
<iframe src="assets/hypothesis-null.html" frameborder="0"></iframe>
<p class="caption"><span class="lab">Figure 9.</span> Empirical null distribution over 10,000 permutations, with the observed difference of 0.149 marked.</p>
</div>

The observed difference is 0.149 with a p-value below 0.001. Sleeping windows sit roughly fifteen percentage points higher in charge than waking ones at the same hour and in the same posture, and across 10,000 permutations no rearrangement of the labels produced a difference that large, so I reject the null at the 0.05 level.

This is evidence against the null, not proof of a mechanism. A permutation test on observational data cannot exclude a common cause, and the most likely candidate is the bedtime routine itself, since plugging in and going to sleep are the same act for most people, which would make battery level a proxy for a habit instead of for sleep. That is acceptable for prediction and is precisely why the model relies on it, but it would not support a causal claim.

Two limitations bound the estimate. The waking group contains only 142 windows after subsampling, because people who report lying down at night are usually asleep, so the waking mean is the noisier half of the comparison. The test also runs only on windows where the label exists, which the preceding section showed skews toward better charged phones. Both considerations argue for treating 0.149 as an effect size with wide error bars instead of a point estimate.

One further observation is that instantaneous charging state proved a substantially weaker signal than accumulated charge. Whether a phone is drawing power at this moment says less than whether it has been connected long enough to fill.

## Framing a Prediction Problem

Given one minute of passive phone telemetry, I predict whether the user would label that window as sleeping, which is a binary classification task. The response variable is `label:SLEEPING` over the 285,254 windows in which it is present and the accelerometer recorded. It is the direct target of everything above, since each preceding section examined what distinguishes sleep from its absence, and this column encodes that distinction.

I evaluate with F1 on the sleeping class. Accuracy is unsuitable because only 29.1% of labeled windows are sleep, so a model that always predicts waking scores 70.9% while being useless. The two error types also carry different costs and both matter, in that a false negative fragments a real night and understates its duration, while a false positive counts an hour of lying in the dark as sleep and overstates it. F1 requires precision and recall to be acceptable simultaneously, which is what a usable sleep summary demands. I report precision, recall, and accuracy alongside it so that the tradeoff remains visible.

Every feature is a passive sensor reading drawn from the same one-minute window and available on the device the moment that window closes, so nothing looks forward and no label enters as an input. I exclude all GPS and microphone features despite their presence in the data and their predictive strength, because the premise of the project is what the remaining sensors can achieve alone. I also exclude every other `label:` column, including `LYING_DOWN`, which was useful for framing the hypothesis test but is itself self-reported and would not exist at prediction time.

The split runs along user boundaries, not rows. A random row split would place minutes from the same night on both sides, allowing a model to score well by memorizing individual habits, and the object here is a tracker that works on a phone it has never encountered. I therefore hold out 25% of users in full, leaving 39 training users covering 208,526 windows and 14 test users covering 76,728, with no data shared between them.

## Baseline Model

### Specification

The baseline uses the two features the exploratory analysis pointed to most directly and nothing else.

| Feature | Type | Encoding |
| --- | --- | --- |
| `acc_std` | Quantitative | Passed through unchanged, since a decision tree splits on thresholds and monotone rescaling would alter nothing |
| `tod` | Nominal | One-hot encoded into six four-hour buckets with unknown categories ignored |

The second is a derived column, not a raw one. Clock hour is cyclic and has no meaningful zero, so supplying it as a number would let the model treat 23:00 and 00:00 as maximally distant. Bucketing into six blocks and one-hot encoding removes the ordering entirely. The encoding is blunt, and replacing it is one of the changes the final model makes.

The estimator is a depth-3 decision tree, kept shallow deliberately, because with two inputs and clear threshold structure in both, additional depth would fit individual quirks that do not transfer to held-out participants. Feature transformation and model fitting occupy a single scikit-learn pipeline.

| Split | F1 | Precision | Recall | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Train | 0.790 | 0.799 | 0.781 | 0.884 |
| Test, unseen users | 0.833 | 0.877 | 0.794 | 0.897 |

### What the model actually learned

The result is partly good. An F1 of 0.833 on fourteen users the model has never seen is genuine, and it stands well above the 0.488 obtained by predicting sleep on every window. Two features and three splits recover most of the available signal, which indicates the problem is easy in the bulk.

Breaking recall down by hour shows that the headline score is misleading. Recall exceeds 0.94 at every hour from midnight through eight in the morning and is exactly zero at every other hour. A depth-3 tree given these two inputs has discovered a single rule, that sleeping means still and between midnight and eight, and consequently misses every sleeping minute reported at eleven at night or after eight in the morning, amounting to 4,491 windows in the test set alone.

Three consequences follow. The model cannot represent sleep onset or wake time, which are the two quantities a sleep tracker exists to report and which fall precisely in the hours where recall vanishes; an overall recall of 0.794 conceals this because the band from midnight to eight contains most of the sleep. Test F1 also exceeds train F1 (0.833 vs 0.790) which does not indicate strong generalization but reflects the held out users sleeping more (32.3% vs 27.9%), inflating F1 on the positive class and marking a single fourteen-user split as a noisy estimate. Finally the two features compensate for each other badly, since stillness cannot separate sleep from a phone on a desk, clock time cannot separate sleep from lying awake at two in the morning, and neither feature knows anything about the preceding twenty minutes, when sleep is sustained stillness, and no single still minute settles it.

## Final Model

### Engineered features

Four additions go in, each aimed at the hours where the baseline scored nothing at all.

Rolling stillness is the mean of `acc_std` over the trailing 30 minutes, computed inside each user so that one participant's night never bleeds into another's. Sleep is sustained stillness, and no single quiet minute settles it, and the baseline had no way to express that because it treated every window as independent.

Time since the phone was last handled counts the minutes since the most recent window with the app in the foreground or a call active, forward-filled within each user and capped at 720. The cap stops the first window of a recording, or any long break in recording, from becoming an arbitrarily large number, and twelve hours is already far beyond the point where the feature discriminates. Falling asleep follows putting the phone down, so this describes the transition into sleep, not the state itself, which is what the failures at eleven at night required.

Cyclic hour replaces the six one-hot buckets with the sine and cosine of the clock angle, placing 23:50 next to 00:10 and dissolving the hard boundary the baseline had built at midnight. The buckets remain in the feature set, so the final model sees a strict superset of the baseline features.

The privacy-safe phone state contributes `battery_level`, `battery_state`, `ringer_mode`, and `screen_brightness`. The hypothesis test established that battery level separates sleep from waking rest at the same hour and in the same posture, which is precisely where clock time and stillness both fail. Quantitative columns are median-imputed and nominal ones are filled with an explicit unknown category before one-hot encoding, since a missing ringer mode means an iPhone, not a silent phone and deserves to be modelled as its own state.

Both temporal features look strictly backwards. A centered window would describe sleep better, because the minutes following a still minute are as informative as the minutes preceding it, but the framing above committed to using only what the device holds at the moment the window closes. A retrospective nightly summary could relax that constraint. A live tracker cannot, so I did not.

### Model selection

The estimator moves to a random forest, because the boundary here is a conjunction of thresholds across several sensors and a single tree has to spend depth re-splitting on the same columns to express it.

The two hyperparameters chosen for tuning before the search was run are maximum depth and minimum samples per leaf. Depth governs how much individual structure the forest can memorize, which is the failure mode that matters when every test user is a stranger, and leaf size governs whether the model can carve out individual single stray minutes. Depth was searched over 8, 14, and unlimited, and leaf size over 1 and 20.

Cross-validation folds are grouped by user. Ordinary k-fold would scatter one person's minutes across every fold and then select whichever setting memorized them best, which is the exact opposite of what the held out-user test set measures.

| max_depth | min_samples_leaf | Grouped CV F1 |
| ---: | ---: | ---: |
| 8 | 1 | 0.7925 |
| 8 | 20 | 0.7932 |
| 14 | 1 | 0.7682 |
| 14 | 20 | 0.7757 |
| None | 1 | 0.7681 |
| None | 20 | 0.7781 |

The search preferred the shallowest depth offered and the larger leaf size. That is what a split grouped by user rewards, since deeper forests memorize individual sleep habits that do not transfer, and it is the reason the grouping matters and is not a technicality.

### Result

| Model | Split | F1 | Precision | Recall | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | Train | 0.790 | 0.799 | 0.781 | 0.884 |
| Baseline | Test, unseen users | 0.833 | 0.877 | 0.794 | 0.897 |
| Final | Train | 0.863 | 0.895 | 0.832 | 0.926 |
| Final | Test, unseen users | 0.870 | 0.925 | 0.822 | 0.921 |

The final model reaches an F1 of 0.870 on the held out users against 0.833 for the baseline, gaining about five points of precision and three of recall at once. The aggregate understates what changed.

<div class="figure" markdown="0">
<iframe src="assets/recall-by-hour.html" frameborder="0"></iframe>
<p class="caption"><span class="lab">Figure 10.</span> Recall on true sleeping windows by hour for both models, evaluated on the fourteen held out users.</p>
</div>

The baseline scored exactly zero at every hour outside the band from midnight to eight. The final model recovers 65% of sleeping windows at eight in the morning, 42% at nine, and 35% at eleven at night, which are the hours where sleep onset and wake time are actually decided and the only hours where a tracker's answer is ever in doubt.

The gain is not free. Recall slips at a few hours well inside the night, from 0.976 to 0.870 at midnight and from 0.960 to 0.908 at seven, because the forest no longer treats the hour alone as sufficient and sometimes waits for the stillness window to agree. Trading a little certainty in the middle of the night for the ability to see the edges of it is the right trade, since the middle of the night was never the part in question.

The model still scores zero from ten in the morning through ten at night, covering roughly 1,900 test windows. Those are daytime naps, and nothing in the privacy-safe feature set distinguishes a nap from a phone resting on a desk. That limitation is real and no aggregate should be allowed to hide it.

| Feature | Importance |
| --- | ---: |
| `acc_std_roll30` | 0.216 |
| `hour_sin` | 0.168 |
| `tod_00-04` | 0.114 |
| `battery_level` | 0.084 |
| `hour_cos` | 0.075 |
| `acc_std` | 0.059 |

The importances support the reasoning behind the additions, not only the outcome. Trailing stillness is the single most important feature, ahead of cyclic hour and the midnight-to-four bucket. Battery level ranks fourth, which is the same signal the hypothesis test isolated earlier, now holding its place inside a model with access to everything else.

## Fairness Analysis

The model leans heavily on clock time, so the question worth asking is whether it works less well for people whose sleep does not fall where the clock expects it.

I split the fourteen held out users by how concentrated their sleep is overnight, measuring for each the share of their sleeping windows falling between ten at night and eight in the morning, then cutting at the median of 0.917. Group X is the seven users below that line, whose sleep is more scattered across the day. Group Y is the seven above it. The evaluation metric stays F1 on the sleeping class, the same one the model was selected on.

The null hypothesis is that the model is fair, meaning its F1 is the same for both groups and any observed difference is due to chance. The alternative is that the model is unfair, with a lower F1 for the irregular schedule group. The test statistic is the difference in F1, irregular minus regular, and the significance level is 0.05.

The permutation shuffles the group label across users, not across windows. Schedule regularity is a property of a person, every window belonging to one user shares it, and shuffling individual windows would break that structure and manufacture a null far narrower than the truth.

| Group | Users | Windows | F1 |
| --- | ---: | ---: | ---: |
| Irregular schedule | 7 | 41,071 | 0.831 |
| Regular schedule | 7 | 35,657 | 0.917 |

<div class="figure" markdown="0">
<iframe src="assets/fairness-users.html" frameborder="0"></iframe>
<p class="caption"><span class="lab">Figure 11.</span> Each held out user, placed by how overnight their sleep is and how well the model scores them. Circle size is the number of labeled sleeping windows.</p>
</div>

Each circle is one held out user, sized by how many sleeping windows they contribute. The two circles sitting near zero on the left and at the bottom right are the participants with 25 and 63 labeled sleeping windows, and they move the group averages far more than their evidence warrants. Any conclusion drawn from fourteen users has to survive those two, which is the real constraint on the test below.

<div class="figure" markdown="0">
<iframe src="assets/fairness-null.html" frameborder="0"></iframe>
<p class="caption"><span class="lab">Figure 12.</span> Empirical null distribution over 10,000 permutations of the group assignment at the user level, with the observed difference marked.</p>
</div>

The observed difference is 0.086 in the direction the alternative predicted, with a p-value of 0.057. That sits just above the 0.05 threshold, so I fail to reject the null hypothesis at the stated level.

Failing to reject is not the same as finding the model fair, and here that distinction carries most of the weight. The unit of resampling is the user, there are only fourteen of them, and the permutation null accordingly has a standard deviation of 0.053. A difference would have to be enormous to clear that bar. Nearly nine points of F1 is substantively large for anyone whose schedule is unusual, and it runs in the direction the model's own construction implies, since cyclic hour and the time-of-day buckets together carry more importance than any feature except trailing stillness.

The honest summary is that the evidence is suggestive but does not reach the threshold I set, and that the study is too small to settle the question. The model is not shown to be fair. Two of the seven irregular users contribute fewer than 100 labeled sleeping windows each, so their individual scores are themselves noisy. Settling this would need a held out set of dozens of users deliberately sampled across schedule types, which sixty participants cannot supply.

## Links

- [Codebase](https://github.com/nahatav/sleep-mode)
- [UCSD ExtraSensory Dataset](http://extrasensory.ucsd.edu/)
- [DSC 80 Course Website](https://dsc80.com/)
