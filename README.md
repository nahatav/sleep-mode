Valmik Nahata. DSC 80 at UCSD, Final Project. Data: the [UCSD ExtraSensory dataset](http://extrasensory.ucsd.edu/).

## Introduction

Your phone can already tell when you are asleep. The question is what it has to look at to do it.

Sleep staging apps ask for the microphone. Commercial trackers ask for location. Both are far more revealing than the thing they are trying to measure, and once granted, neither permission is limited to nighttime. So the question this project is built around is:

**How well can a phone tell that you are asleep using only its dullest signals, meaning motion, battery, ringer, screen, and the clock? No microphone, no GPS.**

If the boring sensors are good enough, then a sleep tracker asking for the invasive ones is asking for something it does not need. That is a claim you can actually test, and the answer decides whether "we need mic access to track your sleep" is a technical constraint or a product decision.

The [UCSD ExtraSensory dataset](http://extrasensory.ucsd.edu/) is well suited to it. It is a year of in-the-wild smartphone and smartwatch telemetry from 60 volunteers, collected on this campus between June 2015 and June 2016. Every row is a single one-minute window described by about 225 pre-computed sensor features, paired with 51 context labels that the users reported about themselves. Crucially it holds both the invasive sensors and the boring ones, so the comparison is possible on identical windows rather than across studies.

I use **377,346 windows from all 60 users**. The full files carry 278 columns; I keep 30, chosen to cover the sensor families that could plausibly say something about sleep, plus the labels I need and two sensors I deliberately exclude from the model later.

| Column | Description |
| --- | --- |
| `uuid` | Anonymized user ID, taken from the filename |
| `timestamp` | Unix time at the start of the one-minute window |
| `raw_acc:magnitude_stats:std` | Standard deviation of accelerometer magnitude in the window. My main stillness measure |
| `raw_acc:magnitude_stats:mean` | Mean accelerometer magnitude. Near 1 g when the phone is at rest |
| `raw_acc:magnitude_stats:value_entropy` | Entropy of the accelerometer magnitude distribution |
| `proc_gyro:magnitude_stats:std` | Rotation variability. Picks up handling that translation misses |
| `lf_measurements:battery_level` | Battery charge as a fraction from 0 to 1 |
| `discrete:battery_state:*` | One-hot family: unplugged, discharging, not charging, charging, full |
| `discrete:ringer_mode:*` | One-hot family: normal, silent with vibrate, silent without vibrate |
| `discrete:app_state:*` | One-hot family: whether the app was foreground, background, or inactive |
| `lf_measurements:screen_brightness` | Screen brightness from 0 to 1 |
| `lf_measurements:light` | Ambient light, already log-scaled |
| `discrete:wifi_status:is_reachable_via_wifi` | Whether the phone had a WiFi connection |
| `discrete:on_the_phone:is_True` | Whether a call was active |
| `label:SLEEPING` | Self-reported sleep. 1, 0, or missing. The response variable |
| `label:LYING_DOWN` | Self-reported posture. Used to separate sleep from awake rest |
| `label:LOC_home` | Self-reported location |
| `location:log_diameter` | Log of the spatial spread of GPS fixes. Held out of the model on purpose |
| `audio_properties:max_abs_value` | Log peak microphone amplitude. Also held out on purpose |
| `raw_magnet:magnitude_stats:std` | Magnetometer variability. Used as a control in the missingness tests |

---

## Data Cleaning and Exploratory Data Analysis

### Cleaning

Seven things needed fixing before any of this was usable.

**1. Sixty files into one frame.** Each user lives in a separate `.csv.gz` named by their anonymized ID, and the ID appears nowhere inside the file. I pulled it from the filename and added it as a `uuid` column so user identity survives the concatenation. Everything downstream depends on this, because the train/test split is by user.

**2. Unix time into local clock time.** `timestamp` is seconds since epoch, which is useless for a sleep question. The study ran at UCSD, so I converted to `America/Los_Angeles`. This is not cosmetic. A daylight-saving-naive conversion shifts half the year by an hour, and an hour is a large error when the whole point is where sleep sits on the clock.

**3. One-hot families back into categorical columns.** The `discrete:` sensors arrive already one-hot encoded, so `battery_state` is spread across six binary columns. That is the wrong shape for both `groupby` and `OneHotEncoder`, and it hides the fact that the categories are mutually exclusive. I collapsed each family into a single nominal column.

**4. The `:missing` indicators are genuinely missing, not a category.** Each family has a `:missing` column. For `ringer_mode` it is set on 58.6% of windows, almost all from iPhones, because iOS does not expose ringer state to third-party apps. That is a device limitation, not a silent phone, so encoding it as "silent" would invent data. I map those windows to `NaN`.

**5. Dropped `lf_measurements:proximity`.** It is exactly 0 in all 377,346 windows. A column with one value cannot carry information.

**6. Left the labels missing.** `label:SLEEPING` is missing in 24.4% of windows. Filling it with 0 would silently assert that every unlabeled window is awake, which is close to the opposite of the truth. The next section is about why those windows are missing, so imputing here would erase the thing worth studying.

**7. Built a night index.** A night crosses midnight, so calendar date splits every sleep episode in two. I shifted each timestamp back 12 hours and took the date, so a "night" runs noon to noon and is named for the evening it starts on. `hours_since_noon` then runs 0 to 24 across that window with no wraparound, which makes sleep timing directly averageable.

I did not touch `lf_measurements:light` or `audio_properties:max_abs_value`, which look like they need a log transform but are already log-scaled at source.

The first few rows of the cleaned frame:

| uuid                                 | datetime                  |    hour |   acc_std |   battery_level | battery_state   |   ringer_mode |   asleep |
|:-------------------------------------|:--------------------------|--------:|----------:|----------------:|:----------------|--------------:|---------:|
| 00EABED2-271D-49D8-B599-1D4A09240601 | 2015-10-05 14:06:01-07:00 | 14.1    |  0.003529 |            0.46 | charging        |           nan |        0 |
| 00EABED2-271D-49D8-B599-1D4A09240601 | 2015-10-05 14:07:01-07:00 | 14.1167 |  0.004172 |            0.46 | charging        |           nan |        0 |
| 00EABED2-271D-49D8-B599-1D4A09240601 | 2015-10-05 14:08:01-07:00 | 14.1333 |  0.003667 |            0.46 | charging        |           nan |        0 |
| 00EABED2-271D-49D8-B599-1D4A09240601 | 2015-10-05 14:09:01-07:00 | 14.15   |  0.003541 |            0.46 | charging        |           nan |        0 |
| 00EABED2-271D-49D8-B599-1D4A09240601 | 2015-10-05 14:10:31-07:00 | 14.1667 |  0.037653 |            0.47 | unplugged       |           nan |        0 |

### Univariate analysis

<iframe src="assets/motion-distribution.html" width="800" height="470" frameborder="0"></iframe>

Accelerometer variability is bimodal on a log scale: a tall narrow mode near 10^-2.8 where the phone is sitting completely still, and a broad low mode near 10^-1.1 where it is being carried or handled, with a clear trough between them. A single threshold on this column therefore separates most windows cleanly, but the still mode holds both a sleeping user and a phone abandoned on a desk, so stillness alone cannot be the whole story.

<iframe src="assets/sleep-duration.html" width="800" height="470" frameborder="0"></iframe>

Across the 194 nights with enough labels to measure, recorded sleep centers on a median of 7.0 hours with a long left tail. Part of that tail is real short nights and part is gaps in recording, since a phone that stops sampling at 3am produces a night that looks identical to an early wake-up.

### Bivariate analysis

<iframe src="assets/sleep-clock.html" width="800" height="470" frameborder="0"></iframe>

Sleep peaks near 90% between 3am and 5am and bottoms out under 2% in the late afternoon, with steep transitions around 11pm and 8am. The daytime floor is not zero, which matters: naps and irregular schedules mean clock time can never fully determine sleep on its own.

The harder comparison is between sleep and awake rest, since lying in bed at 1am looks a lot like sleeping at 1am. Restricting to windows where the user reported lying down between 10pm and 6am holds posture and clock time roughly fixed and asks what is left.

<iframe src="assets/battery-by-state.html" width="800" height="470" frameborder="0"></iframe>

The split is almost entirely in one band. 58% of sleeping windows show a completely full battery against 29% of awake ones, while awake-in-bed windows pile up in the 25% to 75% middle where a phone has been in use all evening. The plausible mechanism is bedtime routine: the phone goes on the charger, tops up, and stays there, while someone awake at 2am is draining it.

### Interesting aggregates

Sleep rate split by both clock time and battery state, over all 285,268 labeled windows:

| tod   |   unplugged |   discharging |   not_charging |   charging |   full |
|:------|------------:|--------------:|---------------:|-----------:|-------:|
| 00-04 |       0.647 |         0.558 |          0.967 |      0.845 |  0.904 |
| 04-08 |       0.471 |         0.564 |          0.708 |      0.791 |  0.907 |
| 08-12 |       0.063 |         0.083 |          0.297 |      0.082 |  0.657 |
| 12-16 |       0.042 |         0.024 |          0     |      0.03  |  0.089 |
| 16-20 |       0.039 |         0.007 |          0     |      0.028 |  0.219 |
| 20-24 |       0.094 |         0.033 |          0.527 |      0.327 |  0.293 |

A full battery beats an unplugged one in every single time bucket, and often by a lot. Between 08:00 and 12:00 the sleep rate goes from 0.06 unplugged to 0.66 full, which is the difference between "almost certainly up" and "probably still in bed". So battery state carries information that clock time does not, which is the entire premise of the model later on.

Actively charging is much less consistent, and in the afternoon it sits slightly below unplugged. What tracks sleep is accumulated charge, not the act of drawing power: a phone plugged in at 2pm is a desk phone, a phone that is full at 6am has been on the nightstand for eight hours.

The `not_charging` column rests on only 3,986 labeled windows spread across six buckets, and its 00:00 to 04:00 cell on 30 of them, so those cells swing hard and should not be read closely.

---

## Assessment of Missingness

### NMAR analysis

I believe `label:SLEEPING` is **MNAR**.

The labels are self-reported through the ExtraSensory app, and a sleeping person cannot report anything. Every label covering sleep is either entered before falling asleep or reconstructed after waking, so the act of being asleep is itself what prevents the label from being recorded. The probability that the value is missing depends on the value it would have taken, which is the definition of MNAR. The 24.4% missing rate is not spread evenly either. It concentrates in long overnight blocks where a user went to bed without setting a label.

The additional data that would make this MAR is the app's own interaction log: when the user was prompted, when the app was foregrounded, when a notification was dismissed. If I knew a window fell in a stretch where the user was never prompted, the missingness would be explained by the prompt schedule rather than by the sleep state, and I could treat it as MAR given prompt history. The public release does not include it.

### Missingness dependency

One methodological note first. The rows are one-minute windows from a continuous recording, so consecutive rows are nearly identical. Treating 377,346 of them as independent draws makes every permutation test return p = 0, including tests on columns with no plausible connection to the outcome. To get an effective sample size closer to the real one, I draw **one window per user per hour**, leaving 7,376 rows. Still large, but no longer dominated by within-session autocorrelation.

**Test 1: does the missingness of `label:SLEEPING` depend on `battery_level`?**

- **Null:** the distribution of battery level is the same whether or not the sleep label is missing.
- **Alternative:** the distributions differ.
- **Test statistic:** absolute difference in mean battery level between the missing and non-missing groups.
- **Significance level:** 0.05.

Observed statistic **0.067**, with mean battery 0.605 where the label is missing against 0.672 where it is present.

<iframe src="assets/missingness-battery.html" width="800" height="470" frameborder="0"></iframe>

The observed gap never appears in 5,000 permutations, so **p < 0.001** and I reject the null. Windows with a missing sleep label sit at a lower battery level. The mechanism is straightforward: a phone running low is a phone whose owner has stopped interacting with the app, and a phone that dies overnight stops collecting labels entirely.

**Test 2: does it depend on `magnet_std`?**

Magnetometer variability measures how much the local magnetic field fluctuated in the window. It responds to nearby metal and electronics. There is no route from that to whether a user remembered to label their sleep, so this is the control.

- **Null:** the distribution of magnetometer variability is the same whether or not the sleep label is missing.
- **Alternative:** the distributions differ.
- **Test statistic:** absolute difference in mean magnetometer variability.
- **Significance level:** 0.05.

Observed statistic **0.392**, **p = 0.284**. Well above 0.05, so I fail to reject the null. The missingness of `label:SLEEPING` does not appear to depend on the magnetic environment.

| column | observed | p-value | n |
| --- | --- | --- | --- |
| `battery_level` | 0.0669 | < 0.001 | 7,369 |
| `magnet_std` | 0.3916 | 0.284 | 6,719 |

Taken together: the sleep label goes missing more often when the battery is low, and just as often regardless of the magnetic environment. That is the pattern you would expect if missingness is driven by whether the user and their phone were in any state to record something. It is also why everything below uses only windows where the label is present, and that filter biases the sample toward better-charged phones. That is a real limitation of the rest of this project, not a footnote.

---

## Hypothesis Testing

The EDA suggested that among people lying in bed at night, the sleepers have fuller batteries. That is worth testing properly, because it is the cheapest possible sleep signal. Battery level requires no permission on any platform, reveals nothing about where you are or what you said, and is already on screen.

Restricting to windows where the user reported lying down between 10pm and 6am holds posture and clock time roughly fixed, so the comparison is between sleep and awake rest rather than between night and day. I use the same one-per-user-hour subsample as above, for the same autocorrelation reason. That leaves 1,274 windows from 52 users.

- **Null hypothesis:** among windows where a user reported lying down between 10pm and 6am, the mean battery level is the same for windows labeled sleeping and windows labeled not sleeping. Any observed difference is due to chance.
- **Alternative hypothesis:** among those windows, the mean battery level is higher when the user reported sleeping.
- **Test statistic:** difference in mean battery level, sleeping minus awake. Signed rather than absolute because the alternative is directional, and a difference in means rather than a TVD or KS statistic because battery level is quantitative and the claim is about level, not shape.
- **Significance level:** 0.05.

<iframe src="assets/hypothesis-null.html" width="800" height="470" frameborder="0"></iframe>

**Observed difference 0.149. p < 0.001.**

Sleeping windows sit about 15 percentage points higher in charge than awake-in-bed windows at the same hour and in the same posture. Across 10,000 permutations, the largest difference produced by chance never reached that value, so I reject the null at the 0.05 level.

This is evidence against the null, not proof of a mechanism. A permutation test on observational data cannot rule out that something else drives both, and the most likely candidate is the bedtime routine itself: plugging in and going to sleep are the same act for most people, so battery level may be standing in for a habit rather than for sleep. That is fine for prediction, and it is exactly why the model leans on it. It would not be fine as a causal claim.

Two limitations worth naming. The awake group is only 142 windows after subsampling, since people who report lying down at night are usually asleep, so the estimate of the awake mean is the noisy half of this comparison. And the whole test runs on windows where the sleep label exists, which the missingness analysis showed skews toward better-charged phones. Both push toward treating 0.149 as an effect size with wide error bars rather than a point estimate.

Worth noting separately that instantaneous charging state was a much weaker signal than accumulated charge. Whether the phone is drawing power right now says less than whether it has been plugged in long enough to fill up.

---

## Framing a Prediction Problem

**Problem.** Given one minute of passive phone telemetry, predict whether the user would label that window as sleeping. This is **binary classification**.

**Response variable.** `label:SLEEPING`, restricted to the 285,254 windows where it is present and the accelerometer recorded. It is the direct target: everything above was about what separates sleep from not-sleep, and this is the column that encodes it.

**Metric.** **F1 on the sleeping class.** Accuracy is a bad choice here because only 29.1% of labeled windows are sleep, so always predicting "awake" scores 70.9% while being useless. The two error types also cost different things and both matter. A false negative chops a real night into fragments and understates sleep duration. A false positive counts an hour of lying in the dark as sleep and overstates it. F1 forces precision and recall to be good at the same time, which is what a usable sleep summary needs. I report precision, recall and accuracy alongside it so the tradeoff stays visible.

**What is known at prediction time.** Every feature is a passive sensor reading from the same one-minute window, available on-device the moment the window closes. Nothing looks forward, and no label is used as an input.

**What I exclude on purpose.** No GPS features and no microphone features, even though both are in the dataset and both are strong predictors. Leaving them out is the premise of the project: the question is what the dull sensors alone can do. I also exclude every other `label:` column, including `LYING_DOWN`, which was useful for framing the hypothesis test but is itself self-reported and would not exist at prediction time.

**Splitting by user, not by row.** A random row split would put minutes from the same night on both sides, and a model could score well by memorizing individual users' habits. Since the point is a tracker that works on a phone it has never seen, I hold out 25% of users entirely. The 39 training users (208,526 windows) and 14 test users (76,728 windows) share no data.

---

## Baseline Model

The baseline uses the two features the EDA pointed at most directly, and nothing else.

| Feature | Type | Encoding |
| --- | --- | --- |
| `acc_std` | Quantitative | Passed through unchanged. A decision tree splits on thresholds, so monotone rescaling would change nothing |
| `tod` | Nominal | One-hot encoded into six four-hour buckets, with `handle_unknown='ignore'` |

`tod` is a derived column, not a raw one. The underlying hour is cyclic and has no meaningful zero, so feeding it in as a number would let the model treat 23:00 and 00:00 as maximally far apart. Bucketing into six blocks and one-hot encoding removes that ordering entirely. It is a blunt encoding, and improving it is the first thing on the list for the final model.

The estimator is a **depth-3 decision tree**, shallow on purpose: with two inputs and clear threshold structure in both, a deeper tree would fit user-specific quirks that will not transfer to held-out users. Feature transformation and model fitting are in a single `sklearn` Pipeline.

| baseline            |    F1 |   precision |   recall |   accuracy |
|:--------------------|------:|------------:|---------:|-----------:|
| train               | 0.790 |       0.799 |    0.781 |      0.884 |
| test (unseen users) | 0.833 |       0.877 |    0.794 |      0.897 |

### Is it good?

Partly. Test F1 of 0.833 on 14 users the model has never seen is a real result, and a long way above the 0.488 you get from predicting sleep every time. Two features and three splits recover most of the signal, which says the problem is genuinely easy in the bulk.

But breaking recall down by hour shows the model is not doing what the headline score suggests. Recall sits above 0.94 at every hour from midnight to 8am, and is **exactly zero at every other hour**. A depth-3 tree with these two inputs has learned one rule: sleeping means still and between midnight and 8am. Every sleeping minute reported at 11pm, and every one reported after 8am, is missed. That is 4,491 windows in the test set alone.

Three problems follow.

**The model cannot represent sleep onset or wake time.** Those are the two numbers a sleep tracker exists to report, and they live precisely in the hours where recall is zero. Overall recall of 0.794 hides this completely, because the 00:00 to 08:00 band is where most of the sleep is.

**Test F1 exceeds train F1**, 0.833 against 0.790. That is not the model generalizing well, it is the held-out users sleeping more, 32.3% against 27.9%, which inflates F1 on the positive class. A single 14-user split is a noisy estimate.

**The two features cover for each other badly.** Stillness cannot tell sleep from a phone on a desk. Clock time cannot tell sleep from lying awake at 2am, and the hypothesis test showed how much is left on the table there. Neither feature knows anything about the last twenty minutes, and sleep is sustained stillness rather than any single still minute.

---

## Final Model

*In progress.* Four additions are planned, aimed squarely at the zero-recall hours above:

1. **Rolling stillness**, the mean of `acc_std` over a centered 31-minute window within each user. A single still minute is ambiguous, half an hour of stillness is not. This is the feature the baseline most obviously lacks, since it currently treats every minute as independent.
2. **Time since the phone was last handled**, in minutes since the most recent foreground or call window. Sleep follows putting the phone down, so this encodes the transition into sleep rather than the state.
3. **Cyclic hour**, replacing the six one-hot buckets with sine and cosine of the clock angle, so 23:50 and 00:10 sit next to each other. The current bucketing is what creates the hard boundary at midnight in the first place.
4. **The privacy-safe phone state**: `battery_level`, `battery_state`, `ringer_mode`, `screen_brightness`. The hypothesis test established that battery level carries real information exactly where clock time and stillness both fail.

The estimator moves to a `RandomForestClassifier`, tuning `max_depth` and `min_samples_leaf` by `GridSearchCV` on the same split and the same metric, so the comparison against the baseline is direct.

---

## Fairness Analysis

*In progress.* The planned comparison is model F1 on users whose sleep is concentrated overnight against users with irregular or daytime schedules, tested by permutation. A model that leans as heavily on clock time as this one does should be expected to fail the second group, and the point of the analysis is to measure by how much.
