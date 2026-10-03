# The Ont3 tag inventory

This document describes what the tagger predicts, how labels from public
corpora with other inventories are brought into it, and how to move a trained
model to a different inventory. Paths are relative to the package root.

## What the model outputs

The tagger reads a text window and assigns every subword token one label from
a BIOES scheme over the 31 Ont3 primary types. BIOES marks a token as the
**B**eginning, **I**nside or **E**nd of a multi-token span, a **S**ingle-token
span, or **O**utside any span. With 31 types this gives 31 x 4 + 1 = 125
output labels. Decoding turns the label sequence back into character spans:

```text
Dr. Lena Okafor moved to Rotterdam on 3 May.
[person_name---]         [locality]    [date]
```

(All examples in these documents are invented.)

A **primary type** is a label with its own span boundary. Ont3 also defines
optional refinements that describe a primary span further (who a person is,
which kind of ID a number is); they are covered in
[Refinement channels](#refinement-channels) and do not change primary
boundaries.

## Primary types

Ont3 is the 29 types of the second-generation inventory in
`scripts/pii_tagset_v2.yaml` plus two reference types. The definitions the
annotation teacher sees, including extent rules, are in
`prompts/pii-label/catalog-ontology-v3-primary-v3.md`; where the two files
differ in wording, that catalog is the Ont3 authority. The family and default
redaction action come from `scripts/pii_tagset_v2.yaml`, which predates the
reference types, and are suggestions for a downstream redactor. The model itself only predicts spans and types.

| Type | Family | Covers | Default action |
|---|---|---|---|
| `person_name` | person | A person's name with all components and an adjacent closed-class honorific (Mr., Dr., Frau) | pseudonym |
| `person_reference` | (Ont3 addition) | A pronoun, role phrase or description that points to a specific person ("the claimant", "her brother") | not assigned |
| `organization` | organization | A named company, institution, agency, department, payment network or card brand | pseudonym |
| `organization_reference` | (Ont3 addition) | A descriptive or pronominal mention of a specific organization ("the bank", "the ministry") | not assigned |
| `demographic_attribute` | attribute | Gender, nationality, occupation, job title or office, language spoken, physical or household description of a person | generalize |
| `protected_attribute` | attribute | GDPR Article 9 categories: ethnic origin, religion, political opinion, union membership, sexual orientation | suppress |
| `health_condition` | attribute | Diagnosis, symptom, medication, allergy, blood group, disability, pregnancy, test or treatment | suppress |
| `age` | attribute | A stated age | generalize |
| `date` | date_time | A calendar date, with or without time | generalize |
| `date_of_birth` | date_time | A date that context identifies as a birth date | generalize |
| `time` | date_time | A clock time or time of day without a date | keep |
| `admin_area` | location | Country, state, province, region (and a time zone standing for one) | keep |
| `locality` | location | City, town, village, district, county | generalize |
| `location` | location | Other places: named buildings, facilities, landmarks, rooms, routes | pseudonym |
| `street_address` | location | The premises line of a postal address | pseudonym |
| `postal_code` | location | A postal or ZIP code | generalize |
| `gps_coordinates` | location | A latitude/longitude pair or other explicit coordinate | generalize |
| `email` | contact | An email address | pseudonym |
| `phone_number` | contact | A telephone or fax number | mask |
| `url` | contact | A web address, URI or social-profile locator | pseudonym |
| `username` | digital | An account handle or screen name | pseudonym |
| `ip_address` | digital | An IPv4 or IPv6 address | generalize |
| `device_identifier` | digital | MAC address, IMEI, advertising ID, user-agent string | suppress |
| `credential` | credential | Password, PIN, API key, session token, recovery code | suppress |
| `government_id` | authority_id | National ID, social-security, passport, driving licence, tax or professional licence number | mask |
| `record_identifier` | org_id | Customer, account, patient, policy, employee, order or case number; vehicle registration | pseudonym |
| `bank_account_number` | financial | Account number, IBAN, masked account digits, crypto wallet address | mask |
| `bank_routing_code` | financial | SWIFT/BIC, ABA routing number, sort code | keep |
| `payment_card_data` | financial | Card number, security code, expiry | mask |
| `monetary_amount` | financial | An amount of money, with its currency inside the span | generalize |
| `quantity` | other | A non-currency, quasi-identifying count or measure (areas, share counts, percentages) | generalize |

### Extent conventions that matter in practice

These rules decide where a span starts and ends. They are the most common
source of disagreement with other corpora.

- **Honorifics stay in the name; titles do not.** In "Prof. Ada Mensah", the
  whole string is `person_name`. In "Senator Ada Mensah", `Senator` is a
  separate `demographic_attribute` and only `Ada Mensah` is `person_name`.
  Many public corpora put such titles inside the person span; see
  [evaluation.md](evaluation.md#title-policy) for how scoring handles this.
- **Organization names exclude a leading article.** "the Harbour Trust"
  yields `organization` over `Harbour Trust`. A determiner-bearing
  description with no name ("the trust") is `organization_reference`.
- **Places inside an organization name stay only if they are part of the
  conventional name** ("University of Valdoria, Eastfield" is one span). An
  adjacent place that merely modifies the name is a separate place span.
- **Currency belongs to the amount.** "EUR 1,200" is one `monetary_amount`;
  a currency word with no number is `O`.
- **Durations are not quantities.** "two years" and "30 minutes" are `O`.
- **References never contain the referent's own name.** In "Ada Mensah, the
  plaintiff's lawyer", `Ada Mensah` is `person_name` and `the plaintiff's
  lawyer` is a separate `person_reference`.

## Reference types

`person_reference` and `organization_reference` let the model mark mentions
that point to a specific person or organization without naming it. They are
useful for discovery (following an entity through a document) and for strict
redaction policies. They are also where annotators disagree most, and where
corpora differ: some label "the defendant" as a person, most label nothing.

For that reason redaction scores in the paper treat references as
**optional**: a missed reference costs nothing, a predicted reference earns
nothing, and a `person_name` or `organization` prediction that exactly covers
a gold reference is neutral. The rule is implemented in
`scripts/pii_reference_projection.py` (`optional_reference_spans`) and
described in [evaluation.md](evaluation.md#optional-references). It changes
scoring only; the model still predicts reference types.

## Native source labels and the many-to-many map

Public NER corpora use their own, usually coarser, label sets. Their labels
are called **native source labels** here: `LOC`, `PER` and `ORG` in OpenNER
and AQMAR; `ADDRESS`, `AMOUNT`, `DATE`, `PERSON` and others in MAPA; `GPE`,
`FAC`, `OCC` and others in Wojood.

A native label often corresponds to several Ont3 types. OpenNER's `LOC` may
be a country (`admin_area`), a city (`locality`) or a building (`location`).
Forcing one choice would teach the model a wrong type for many spans. Instead
the map lists every Ont3 type a native label may legitimately be, and training
accepts any of them.

The map is `research/pii/frontier/evidence/four-corpus-v1/candidate-map.json`:

- `source_labels[corpus][label]` names an intermediate node, for example
  `openner_core__LOC`.
- `v1_to_v2[node].accepted` lists the Ont3 types that node may be;
  `fallback` names the single type used where one type must be chosen.
- Nodes named after an Ont3 type (`person_name`, `date`, ...) map to
  themselves, so rows already labeled in Ont3 pass through unchanged.
- `ontology.primary_types` is the 31-type head inventory.

| Corpus | Native label | Accepted Ont3 types |
|---|---|---|
| OpenNER, AQMAR | `PER` | person_name |
| OpenNER, AQMAR | `ORG` | organization |
| OpenNER, AQMAR | `LOC` | admin_area, locality, location, street_address, postal_code, gps_coordinates |
| MAPA | `PERSON` | person_name |
| MAPA | `ORGANISATION` | organization |
| MAPA | `ADDRESS` | the six place types above |
| MAPA | `DATE` | date, date_of_birth |
| MAPA | `TIME` | time |
| MAPA | `AMOUNT` | monetary_amount, quantity |
| Wojood | `PERS` | person_name |
| Wojood | `ORG` | organization |
| Wojood | `GPE`, `LOC`, `FAC` | the six place types above |
| Wojood | `OCC`, `LANGUAGE` | demographic_attribute |
| Wojood | `WEBSITE` | url |
| Wojood | `DATE` | date, date_of_birth |
| Wojood | `TIME` | time |
| Wojood | `MONEY` | monetary_amount |

Wojood labels outside this list (for example `EVENT`, `NORP`, `PRODUCT`) are
dropped at conversion; see the ignored-label list in
`scripts/pii_onboard_sources.py`.

**How the loss uses an accepted set.** For a token inside a `LOC` span, the
training target is not one label but a set, for example
{`B-admin_area`, `B-locality`, `B-location`, ...}. The loss is the negative
log of the total probability the model puts on that set. The model is
rewarded for picking any accepted type, and it learns which one from context
and from rows where the type is labeled directly. It is not rewarded for
`person_name`, and it is not rewarded for a wrong boundary: the B/I/E/S
position must still be right.

**What an accepted set does not say.** A positive mapping tells the model what
a labeled span may be. It says nothing about what the corpus left unlabeled.
Whether an unlabeled token is a trustworthy `O` depends on the corpus's
annotation coverage, which is a separate record
(`research/pii/frontier/evidence/four-corpus-v1/negative-coverage-v1.json`,
explained in [data.md](data.md#complete-supervision-with-unknown-types)).

## The mapped single head

O4 has one primary output layer (a "head") over the 31 Ont3 types. Rows labeled in
Ont3 supervise it directly. Rows labeled with native source labels supervise
the same head through the accepted sets above. This arrangement is called the
**mapped single head**. There is no separate head per corpus, and nothing of
the native inventories survives in the model's output.

In the trainer (`scripts/pii_encoder_train.py`) this is the combination
`--dual-head-map MAP --dual-head-mapped-single-head
--dual-head-old-weight-schedule constant:0`. The option names come from the
more general two-head mechanism described in
[Migrating to a new inventory](#migrating-to-a-new-inventory); with these
settings only the Ont3 head exists and receives all supervision. The
mapped-single-head mode continues an existing Ont3 head and never creates
one, which is why a fresh fit first creates the head with one native-label
update ([training.md](training.md#recipes)).

The consequence for users: the model has learned publisher conventions only
where they agree with Ont3 or where the map absorbs the difference. Extent
conventions the map cannot express (for example titles inside MAPA person
spans) need separate handling in the data
([data.md](data.md#mapa-titles)).

## Refinement channels

Ont3 includes refinements that attach to a primary span. They are auxiliary
outputs and are independent of BIOES.

**Bernoulli predicates.** `scripts/pii_ont3_bernoulli_predicate_channels_v1.json`
defines six yes/no channels for `person_name` and `person_reference` spans:
`care_provider`, `patient`, `family_member`, `witness_or_bystander`,
`investigator_or_law_enforcement` and `legal_professional`. Several or none
may be true. A token is positive when it overlaps a positive character
extent, known-negative when it lies inside an explicitly labeled span of an
applicable type, and unknown (no loss) otherwise.

**Categorical subclass families.** `scripts/pii_subclass_families_v3.json`
defines one-of-N choices conditioned on the primary type:

| Family | Applies to | Outcomes |
|---|---|---|
| `name_component` | person_name (sub-spans) | given, middle, family name |
| `government_id_kind` | government_id | passport, national ID, US SSN, tax ID, driver licence, professional licence, other |
| `place_coarseness` | admin_area, locality | country, first- or second-order admin area, locality, sub-locality |
| `jurisdiction_nation` | government_id, admin_area, locality | one of 41 countries |
| `currency_identity` | monetary_amount, bank_account_number | one of 18 currencies |

Each family also has an outcome `Q`, meaning reliably none of the named
outcomes. A missing annotation is unknown and masked, never read as `Q`.

**Status in O4.** O4 was trained with the predicate head present but all six
predicate channels given zero objective weight, and with the subclass
objective at weight 1.0 where rows carry subclass annotations. Coverage of
these annotations in the training data is partial, and the paper reports no
refinement quality. Treat the refinement outputs as an available mechanism,
not a validated feature.

The practical use of refinements is per-type customization without changing
primary boundaries: for example redacting only care-provider names, or
generalizing a `locality` to its country.

## Migrating to a new inventory

A trained tagger can be moved to a different label set in several ways. They
differ in how much retraining they need and whether old labeled data can
still be used.

**1. Project the output.** If the new inventory is coarser or a subset,
rewrite predictions after decoding: merge types (all six place types into
`LOCATION`), drop types, or change the redaction action per type. No training
is needed. The evaluation code uses the same idea to compare models with
different inventories (`project()` in
`research/pii/frontier/evidence/human-gold-v1/score.py`).

**2. Supervise through a map.** If the new data uses a different inventory
and the model's head stays Ont3, write a map in the `candidate-map.json`
format from each new label to its accepted Ont3 types and continue training
in mapped-single-head mode. This is what O4 did for the four public corpora.

**3. Fade from the old head to a new head.** If the target inventory itself
changes, the trainer can add a second head over the new inventory and train
both from every row: each row supervises its own head directly and the other
head through the map's accepted sets. The relevant options of
`scripts/pii_encoder_train.py`:

| Option | Effect |
|---|---|
| `--dual-head-map MAP` | Without `--dual-head-mapped-single-head`, adds the second head |
| `--dual-head-old-weight-schedule linear:1.0:0.0[:SPAN]` | Moves objective weight from the old head to the new one over the run; `SPAN` finishes the handover in that leading fraction of the steps |
| `--dual-head-init {projected,fallback,fresh}` | Initializes the new head by projecting the old head's weights through the accepted sets, by averaging over fallback edges, or randomly |
| `--dual-head-keep-old-head` | Keeps the old head after its weight reaches zero; by default it is removed, leaving a single-head tagger |
| `--dual-head-restart-lr-after-transition` | Gives the new-head-only phase its own warmup and decay |
| `--dual-head-eval-weight W` | Fixes the head blend used for validation loss |

This fading mechanism is available in the shipped trainer. It is not the
transition used by O4, and the paper reports no result for it on this
inventory.

**4. Start a new head.** With data labeled natively in the new inventory,
`--native-new-label-space` trains a head whose `labels.json` is the new type
list, with no map and no second head. The `o4-fresh` recipe uses this for its
one-update head initialization.

Whichever route you take, keep evaluation in a fixed view: score old and new
models on the same rows through the same projection, so that a change in
inventory is not mistaken for a change in quality.

## Files

| File | Content |
|---|---|
| `prompts/pii-label/catalog-ontology-v3-primary-v3.md` | Ont3 type definitions and extent rules used for annotation |
| `scripts/pii_tagset_v2.yaml` | 29-type inventory with families, default actions and source projections; loaded by `scripts/pii_ontology_v2.py` |
| `scripts/pii_tagset.yaml` | Older unified inventory used by the corpus assembler for native-label conversion |
| `research/pii/frontier/evidence/four-corpus-v1/candidate-map.json` | Native label to accepted Ont3 types |
| `research/pii/frontier/evidence/four-corpus-v1/negative-coverage-v1.json` | Ont3 types each corpus annotates exhaustively |
| `scripts/pii_ont3_bernoulli_predicate_channels_v1.json` | Predicate channel definitions |
| `scripts/pii_subclass_families_v3.json` | Subclass family definitions and training contract |
| `scripts/pii_reference_projection.py` | Optional-reference scoring rule |
