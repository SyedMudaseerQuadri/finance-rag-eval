# Sample answers (dense top-5, generator qwen2.5:3b)

First 5 of the 30 evaluated questions, unedited. Citations are passage IDs from the retrieved set. Judge verdicts come from a small local model and are noisy.

## Q: Most effective Fundamental Analysis indicators for market entry

**Answer:** [81655] Fundamental Analysis can be used to help you determine what to buy, but they won't give you an entry signal for when to buy. Technical Analysis can be used to help you determine when to buy, and can give you entry signals for when to buy. There are many Technical Indicator which can be used as an entry signal, from as simple as the price crossing above a moving average line and then selling when the price crosses back below the moving average line, to as complicated as using a combination of indicators to all line up for an entry signal to be valid.

- Cited: ['81655'] | invalid: [] | validator ok: True
- Judge (qwen3.5:4b): **unsupported** - The answer fails to provide any specific Fundamental Analysis indicators because the provided passages explicitly state that Fundamental Analysis does not offer entry signals.
- Retrieved IDs: ['81655', '425020', '367391', '96910', '224695']

## Q: Specifically when do options expire?

**Answer:** [581672] Regular equity monthly options expire on the 3rd Friday of every month. [358492] Equity options expire the Saturday after the 3rd Friday of the month. [293605] E-mini S&P500 options expire on the 3rd Friday. [116436] Options expired on the 3rd Wednesday of Mar, Jun, Sep, Dec traditionally. [463254] Options expire when they expire, no specific date mentioned. [581672] Specific to monthly options, last trading time is 4 PM Eastern time on the 3rd Friday. [293605] E-mini S&P500 options expire on the 3rd Friday. Therefore, options can expire on the 3rd Friday or the Saturday after the 3rd Friday of the month.

- Cited: ['581672', '358492', '293605', '116436', '463254'] | invalid: [] | validator ok: True
- Judge (qwen3.5:4b): **partially** - The answer is partially supported because it correctly cites specific expiration rules from the provided passages for different option types, but it includes a claim from passage [463254] that contradicts the other factual statements and offers no specific date.
- Retrieved IDs: ['116436', '358492', '581672', '293605', '463254']

## Q: How to contribute to Roth IRA when income is at the maximum limit & you have employer-sponsored 401k plans?

**Answer:** [441632] [360533] When income is at the maximum limit and you have employer-sponsored 401k plans, you can contribute up to $53,000 including employer contributions. However, you can only contribute $5500 total across IRA and Roth IRA. Once you max out your 401k, you can contribute up to $5500 to Roth IRA. If you have a poor employer match, you can contribute more but it would go in as after tax dollars and not claim the tax deduction.

- Cited: ['441632', '360533'] | invalid: [] | validator ok: True
- Judge (qwen3.5:4b): **unsupported** - The answer incorrectly states that one can contribute up to $53,000 to a Roth IRA when income is at the maximum limit, whereas the passages clarify that the total contribution limit for all IRAs combined is only $5,500 and that employer contributions do not count toward this specific limit.
- Retrieved IDs: ['441632', '399543', '290105', '360533', '480036']

## Q: How to donate to charity that will make a difference?

**Answer:** [174033] [319458] [379353] [326167] [8060] To donate to charity that makes a difference, consider using services like CAF (Charities Aid Foundation) for anonymous donations with Gift Aid. Alternatively, use intermediary services like BT MyDonate or The Big Give. For cost-effectiveness, look into services like Give Well which provides median estimates of top charities' cost-effectiveness. Directly support causes you care about, like visiting charities personally. Donating to charities like Veterans Centers or Schools for the Blind can also make a direct impact.

- Cited: ['174033', '319458', '379353', '326167', '8060'] | invalid: [] | validator ok: True
- Judge (qwen3.5:4b): **unsupported** - The answer incorrectly attributes the CAF/Gift Aid information to the US context while the passages only provide this detail for the UK.
- Retrieved IDs: ['174033', '319458', '379353', '326167', '8060']

## Q: What are some well known or well regarded arguments against investing?

**Answer:** [427592] Well-regarded arguments against investing include priorities questioning if you might have something better to do than investing, and the parable of the rich man who thinks about building more barns to store his fruits, suggesting investing may not be the best use of excess cash.

- Cited: ['427592'] | invalid: [] | validator ok: True
- Judge (qwen3.5:4b): **supported** - The answer accurately reflects the content of passage [427592], which presents arguments against prioritizing investment over other uses of time and resources, including the parable of the rich man.
- Retrieved IDs: ['427592', '541302', '378161', '232114', '441603']
