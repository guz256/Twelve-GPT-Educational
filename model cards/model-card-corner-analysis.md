# Model card for Corner Analysis Wordalisation

The Corner Analysis Wordalisation is implemented within the [TwelveGPT Educational framework](https://github.com/soccermatics/twelve-gpt-educational) and is intended as an illustration of how set-piece event data can be converted into concise tactical text. This work is a derivative of the full [Twelve GPT product](https://twelve.football) and follows the same educational philosophy: transparent, data-grounded, interpretable outputs.

This model card follows the structure used in the existing Football Scout model card and adapts it to team-level corner-kick analysis.

Jump to section:

- [Intended use](#intended-use)
- [Factors](#factors)
- [Dataset](#dataset)
- [Model](#model)
- [Evaluation](#evaluation)
- [Ethical considerations](#ethical-considerations)
- [Caveats and recommendations](#caveats-and-recommendations)

## Intended use

The *primary use case* of this wordalisation is to translate team corner metrics into interpretable tactical descriptions. It is designed to support:

- understanding offensive corner style,
- comparing teams in set-piece behaviour,
- assisting tactical analysis and scouting workflows.

Unlike player-scouting wordalisations, the focus here is not individual talent assessment. The emphasis is on *collective set-piece structure* and team execution patterns.

Professional decision-making contexts are possible but remain *out of scope* for this educational version. The tool should not be used as a standalone decision system for recruitment, match strategy, or financial commitments.

## Factors

The Corner Analysis Wordalisation describes teams using the following tactical factors:

- target zone tendency (near-post vs far-post),
- execution mode (short vs direct),
- delivery profile (open-foot vs closed-foot),
- aerial threat profile,
- threat style (phase 1 vs phase 2),
- post-corner transition risk.

An important design principle is that the model mainly describes *style of execution* rather than absolute team quality. In other words, it explains how teams tend to attack corners and the tactical trade-offs that follow.

## Dataset

The dataset is built from corner-related event data and transformed into *team-level aggregated metrics*. The model does not directly describe isolated individual events; instead, it relies on distributions and rates per team.

Example features include:

- target distribution (near / central / far),
- percentage of short corners,
- delivery type distribution (inswing / outswing),
- expected threat proxies from phase 1 and phase 2 actions,
- aerial threat proxies (including HOPS-derived signals when available),
- transition exposure after possession loss from corners.

Features are normalised across teams to support comparability. This allows the same team profile chart to position all teams in one shared reference distribution.

## Model

### Quantitative model

The quantitative layer constructs tactical qualities (`q_*`) from aggregated corner metrics. These qualities are interpretable dimensions of corner behaviour, including:

- `q_target_zone`
- `q_shot_goal_threat`
- `q_threat_style`
- `q_short_vs_direct`
- `q_open_vs_closed`
- `q_header_threat`
- `q_transition_risk`

The core idea is to compress raw event data into a small set of interpretable tactical dimensions.

### Normative model

The normative layer maps each tactical dimension to football-language interpretation. Higher or lower values are translated into tendencies (for example, clear preference, limited use, or balanced profile), while avoiding technical discussion of formulas in user-facing output.

As in the Football Scout approach, the language design prioritises:

- explicit strengths and weaknesses,
- tactical trade-offs,
- concise interpretation over exhaustive metric listing.

Average or neutral signals may be mentioned only when they are relevant to contextual understanding.

### Language model

The language model receives structured context (team metrics, quality profile, and league reference tables) and generates short tactical summaries.

Typical output structure is:

1. style overview,
2. key strengths,
3. limitations and trade-offs,
4. comparative context versus other teams.

The system supports OpenAI/Azure-based chat models and Gemini-based chat models, depending on local configuration.

## Evaluation

Evaluation in this educational version is primarily qualitative. We assess:

- whether the generated text matches the tactical profile in the data,
- whether responses are consistent across repeated prompts,
- whether analysts can interpret outputs quickly and correctly.

During development, prompt iteration and manual review were used to improve factual alignment and reduce unsupported claims.

## Ethical considerations

Compared with player-level scouting, team-level corner wordalisation has lower reputational risk because it does not directly evaluate individual people.

However, relevant risks remain:

- reduction of complex tactical behaviour into a compact textual summary,
- potential overconfidence in model outputs,
- dependence on data quality and tagging completeness.

The tool should be treated as an analytical aid, not as an autonomous evaluator.

## Caveats and recommendations

This model does not capture all tactical details of set-piece behaviour. In particular, current event data may not fully represent:

- blocker actions and screening patterns,
- off-ball movement timing and coordination,
- individual execution quality per delivery,
- some second-ball and recovery micro-patterns.

Output quality also depends on corner sample size. Small samples may produce unstable profiles. The wordalisation should therefore be used together with video analysis and domain expertise, not as a replacement for either.
