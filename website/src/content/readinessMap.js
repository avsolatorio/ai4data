// Maps the dimensions of the AI-readiness assessment framework to the
// program's workstreams, tools, references, and cookbook recipes.
//
// Question numbering follows the assessment form in the reference guide
// (draft v0.0.8, June 2026). Only question ids and short topics are listed;
// the full wording is in the framework. Update FRAMEWORK_VERSION and the
// question lists together when a new draft is published.

export const FRAMEWORK_VERSION = 'reference guide draft v0.0.8 (June 2026)';

// Kinds of support the program offers, used as the columns of the matrix.
export const supportKinds = [
  {id: 'tools', label: 'Open-source tools', short: 'Tools'},
  {id: 'methods', label: 'Methods and research', short: 'Methods'},
  {id: 'recipes', label: 'Cookbook recipes', short: 'Recipes'},
  {id: 'reference', label: 'Reference implementations', short: 'Reference'},
  {id: 'guidance', label: 'Standards and guidance', short: 'Guidance'},
];

// Program resources referenced below. `to` may be internal or external.
export const resources = {
  metadataSchemas: {
    title: 'World Bank metadata schemas',
    to: 'https://github.com/worldbank/metadata-schemas',
    kind: 'guidance',
    note: 'JSON Schema for indicators, microdata (DDI), geospatial (ISO 19115), documents, tables, images, scripts',
  },
  metadataEditor: {
    title: 'Metadata Editor',
    to: 'https://github.com/worldbank/metadata-editor',
    kind: 'tools',
    note: 'Templates, validation, export to SDMX, schema.org, Croissant, DCAT; publishes to NADA',
  },
  nada: {
    title: 'NADA catalog',
    to: 'https://nada.ihsn.org/',
    kind: 'tools',
    note: 'schema.org on every page, keyword and semantic search, REST API, MCP interface, managed access',
  },
  metadataQuality: {
    title: 'Generative AI for Metadata Quality',
    to: '/docs/metadata-quality/generative-ai-for-metadata-quality',
    kind: 'methods',
    note: 'Scores completeness, semantic alignment, specificity, consistency',
  },
  metadataReviewer: {
    title: 'Metadata Reviewer',
    to: '/docs/metadata-reviewer/overview',
    kind: 'tools',
    note: 'Agentic review with a curator review board',
  },
  metadataAugmentation: {
    title: 'Metadata Augmentation',
    to: '/docs/metadata-augmentation/',
    kind: 'tools',
    note: 'DDI-style variable groups from data dictionaries',
  },
  anomaly: {
    title: 'Anomaly Detection and Explanation',
    to: '/docs/anomaly-detection/',
    kind: 'tools',
    note: 'Flags unusual values and classifies the cause with cited evidence',
  },
  discoverability: {
    title: 'Data Discoverability',
    to: '/docs/data-discoverability/',
    kind: 'methods',
    note: 'Semantic search over catalog records',
  },
  pift: {
    title: 'PI-FT embedding toolkit',
    to: '/pift-toolkit/pipeline',
    kind: 'tools',
    note: 'Fine-tunes embedding models on structured records; reports Recall@k, MRR, nDCG',
  },
  mcpDocs: {
    title: 'Model Context Protocol for statistics',
    to: '/docs/mcp/',
    kind: 'methods',
    note: 'Architecture and tool design for an MCP server over official statistics',
  },
  data360Mcp: {
    title: 'Data360 MCP server',
    to: 'https://github.com/worldbank/data360-mcp',
    kind: 'reference',
    note: 'Search, metadata, data, and analysis tools over the Data360 API',
  },
  wdiSdmx: {
    title: 'WDI SDMX API',
    to: 'https://datahelpdesk.worldbank.org/knowledgebase/articles/1886701-sdmx-api-queries',
    kind: 'reference',
    note: 'World Development Indicators through the SDMX 2.1 REST interface',
  },
  data360Api: {
    title: 'Data360 API',
    to: 'https://data360.worldbank.org/en/api',
    kind: 'reference',
    note: 'Indicator search, metadata, disaggregation, and data endpoints',
  },
  microdataLibrary: {
    title: 'Microdata Library',
    to: 'https://microdata.worldbank.org/',
    kind: 'reference',
    note: 'NADA catalog with DDI records and DataCite DOIs',
  },
  pcn: {
    title: 'Proof-Carrying Numbers',
    to: 'https://arxiv.org/abs/2509.06902',
    kind: 'methods',
    note: 'Each number in a generated answer is verified against the record or flagged',
  },
  dataUse: {
    title: 'Monitoring of Data Use',
    to: '/docs/data_use/',
    kind: 'tools',
    note: 'Dataset mention extraction and harmonization',
  },
  synthetic: {
    title: 'REaLTabFormer synthetic data',
    to: 'https://github.com/avsolatorio/RealTabFormer',
    kind: 'tools',
    note: 'Synthetic tabular and relational data for sharing and research',
  },
  inclusive: {
    title: 'Efficient and Inclusive AI',
    to: '/docs/inclusive-ai/',
    kind: 'methods',
    note: 'Model size by task, batch processing, open-weight and local models',
  },
  dataSnapshots: {
    title: 'Data Snapshots',
    to: 'https://arxiv.org/abs/2606.06242',
    kind: 'methods',
    note: 'Layout detection to extract tables and figures from documents',
  },
  knowledgeGraphs: {
    title: 'Ontologies and knowledge graphs',
    kind: 'methods',
    note: 'Workstream: concept links across catalogs',
  },
  standardsWs: {
    title: 'AI-ready metadata and standards',
    kind: 'guidance',
    note: 'Workstream: standards adoption guidance for AI-ready data',
  },
  questionBank: {
    title: 'Global Question Bank',
    kind: 'guidance',
    note: 'Workstream: reusable questions and concepts across surveys',
  },
  responsibleAi: {
    title: 'Responsible AI guidance',
    kind: 'guidance',
    note: 'Workstream: AI use in official statistics under the Fundamental Principles',
  },
  classification: {
    title: 'Statistical classification and coding',
    kind: 'methods',
    note: 'Workstream: AI-assisted coding against standard classifications with human review',
  },
  smallAgentic: {
    title: 'Small and Agentic AI',
    kind: 'methods',
    note: 'Workstream: small models and agentic workflows for statistical tasks',
  },
  openSource: {
    title: 'ai4data repository',
    to: 'https://github.com/worldbank/ai4data',
    kind: 'tools',
    note: 'All program methods and software published as open source (MIT)',
  },
  partnerships: {
    title: 'Partnerships',
    to: '/docs/partnerships/',
    kind: 'guidance',
    note: 'Collaboration across the development data and AI communities',
  },
  // Cookbook
  cb: {
    title: 'Practical Guide to AI-Ready Data Dissemination',
    to: '/cookbook/ai-ready-dissemination/',
    kind: 'recipes',
    note: 'Nine chapters with recipes, profiles, and checklists',
  },
  cbStart: {
    title: 'Cookbook self-assessment',
    to: '/cookbook/ai-ready-dissemination/start-here',
    kind: 'recipes',
    note: 'Five minutes; places an organization at a level per chapter',
  },
  cbStandards: {
    title: 'Standards used in the cookbook',
    to: '/cookbook/ai-ready-dissemination/standards',
    kind: 'guidance',
    note: 'SDMX, DDI, DCAT, DataCite, ISO 19115, and the World Bank schemas, by chapter',
  },
  cbDataTypes: {
    title: 'Cookbook: by data type',
    to: '/cookbook/ai-ready-dissemination/data-types',
    kind: 'recipes',
    note: 'Indicators, microdata, geospatial, documents, tables',
  },
};

const CB = '/cookbook/ai-ready-dissemination/';

export const dimensions = [
  // ---------------------------------------------------------------- Pillar I
  {
    id: '1.1',
    pillar: 1,
    name: 'Strategy and governance',
    short: 'Strategy',
    coverage: 'partial',
    summary:
      'Strategy, oversight bodies, risk management, ethics principles, and change management are institutional decisions. The program contributes guidance and templates for an AI-use policy, a component register, and human review, which are the instruments that questions 1.1.3 and 1.1.4 ask about.',
    questions: [
      {id: '1.1.1', topic: 'AI strategy'},
      {id: '1.1.2', topic: 'Governance and oversight'},
      {id: '1.1.3', topic: 'AI risk management', covered: true},
      {id: '1.1.4', topic: 'Ethical guidelines and principles', covered: true},
      {id: '1.1.5', topic: 'Change management strategy'},
    ],
    support: {tools: 0, methods: 0, recipes: 1, reference: 0, guidance: 1},
    resources: ['responsibleAi'],
    cookbook: [{to: `${CB}govern`, label: 'Chapter 8: Governance and responsible operation (policy template, component register, injection tests)'}],
    steps: [
      {from: 'A', to: 'B', text: 'Adopt the one-page AI-use policy template and name an owner for each AI component.'},
      {from: 'B', to: 'C', text: 'Keep the component register with review dates; require human approval for generated metadata and text.'},
      {from: 'C', to: 'D', text: 'Log model inputs and outputs, run prompt-injection tests in the evaluation suite, and keep an exit plan per provider.'},
    ],
    evidence: ['AI-use policy', 'AI component register with owners and review dates'],
  },
  {
    id: '1.2',
    pillar: 1,
    name: 'Legal environment',
    short: 'Legal',
    coverage: 'partial',
    summary:
      'Legal frameworks, external data access, and accountability provisions are outside the program\'s scope. For question 1.2.6 the program serves as a working example: every method and tool is published under an open-source licence, and the cookbook describes what an organization needs in place to do the same.',
    questions: [
      {id: '1.2.1', topic: 'Legal and policy frameworks for AI'},
      {id: '1.2.2', topic: 'Compliance with external regulations'},
      {id: '1.2.3', topic: 'Legal enablement for public data access'},
      {id: '1.2.4', topic: 'Legal enablement for private data access'},
      {id: '1.2.5', topic: 'Third-party data and intellectual property'},
      {id: '1.2.6', topic: 'Open-source publication of code', covered: true},
      {id: '1.2.7', topic: 'Accountability for AI-generated outputs'},
    ],
    support: {tools: 1, methods: 0, recipes: 0, reference: 1, guidance: 0},
    resources: ['openSource'],
    cookbook: [],
    steps: null,
    evidence: ['Open-source repository with licence, for organizations that adapt program code'],
  },
  {
    id: '1.3',
    pillar: 1,
    name: 'Skills and capacity',
    short: 'Skills',
    coverage: 'partial',
    summary:
      'Workforce planning, recruitment, and incentives rest with the organization. The program supports AI literacy and skills development through documentation that explains each method, runnable notebooks, and a cookbook that a technical team can work through recipe by recipe.',
    questions: [
      {id: '1.3.1', topic: 'AI literacy across the workforce', covered: true},
      {id: '1.3.2', topic: 'Specialized AI expertise'},
      {id: '1.3.3', topic: 'Multidisciplinary team formation'},
      {id: '1.3.4', topic: 'Skills development and external knowledge', covered: true},
      {id: '1.3.5', topic: 'Incentives for adoption of AI tools'},
    ],
    support: {tools: 0, methods: 1, recipes: 2, reference: 0, guidance: 1},
    resources: ['cb', 'cbStart', 'inclusive', 'partnerships'],
    cookbook: [
      {to: `${CB}`, label: 'The cookbook as training material: recipes with code, scope notes, and tests'},
      {to: `${CB}start-here`, label: 'Self-assessment to place a team and choose a reading order'},
    ],
    steps: [
      {from: 'A', to: 'B', text: 'Use the documentation and notebooks as the first exposure to AI methods on familiar statistical tasks.'},
      {from: 'B', to: 'C', text: 'Work through the cookbook chapters as a structured learning path for a small technical core.'},
      {from: 'C', to: 'D', text: 'Contribute recipes and examples back; join the program partnerships for external knowledge.'},
    ],
    evidence: ['Completed recipes and evaluation results as records of skills applied'],
  },
  {
    id: '1.4',
    pillar: 1,
    name: 'Technical environment',
    short: 'Technology',
    coverage: 'partial',
    summary:
      'Connectivity, cloud services, computing, environments, and security are infrastructure decisions outside the program\'s scope. The program contributes to three questions: interoperability standards (1.4.7) through the metadata schemas, privacy-preserving techniques (1.4.10) through synthetic data, and evaluation infrastructure (1.4.11) through its evaluation methods and the cookbook\'s test suite.',
    questions: [
      {id: '1.4.1', topic: 'Electricity and connectivity'},
      {id: '1.4.2', topic: 'Cloud strategy'},
      {id: '1.4.3', topic: 'Data infrastructure for AI'},
      {id: '1.4.4', topic: 'Computing resources'},
      {id: '1.4.5', topic: 'Development, testing, staging'},
      {id: '1.4.6', topic: 'IT security for AI'},
      {id: '1.4.7', topic: 'Standard tools and interoperability standards', covered: true},
      {id: '1.4.8', topic: 'AI productivity tools'},
      {id: '1.4.9', topic: 'Adoption of AI tools in practice'},
      {id: '1.4.10', topic: 'Privacy-preserving AI techniques', covered: true},
      {id: '1.4.11', topic: 'Model evaluation and benchmarking', covered: true},
    ],
    support: {tools: 2, methods: 2, recipes: 2, reference: 0, guidance: 1},
    resources: ['metadataSchemas', 'synthetic', 'pift', 'pcn', 'inclusive', 'smallAgentic'],
    cookbook: [
      {to: `${CB}evaluate`, label: 'Chapter 6: Evaluation (question sets, Recall@k and MRR, answer-level tests)'},
      {to: `${CB}sustain`, label: 'Chapter 9: Sustainability and maintenance (smallest sufficient model, switch-off test, cost per query)'},
    ],
    steps: [
      {from: 'A', to: 'B', text: 'Adopt the metadata schemas as the interoperability baseline and write a first known-item question set.'},
      {from: 'B', to: 'C', text: 'Automate retrieval scoring per language; use synthetic data where confidential microdata cannot be shared.'},
      {from: 'C', to: 'D', text: 'Run answer-level tests (numeric accuracy, citations, refusals) before every model or index change; benchmark small models against large ones.'},
    ],
    evidence: ['Evaluation suite with versioned question set and scores per run', 'Synthetic data release notes with disclosure risk measures'],
  },
  {
    id: '1.5',
    pillar: 1,
    name: 'Funding and resources',
    short: 'Funding',
    coverage: 'none',
    summary:
      'Funding structure, lifecycle coverage, donor management, return tracking, and leadership sponsorship rest entirely with the organization. The cookbook\'s cost-per-query recipe supplies one input to question 1.5.4, and open-source tools reduce licence costs; the program has no component for this dimension.',
    questions: [
      {id: '1.5.1', topic: 'Structure of AI funding'},
      {id: '1.5.2', topic: 'Lifecycle funding coverage'},
      {id: '1.5.3', topic: 'External and donor funding'},
      {id: '1.5.4', topic: 'Tracking value and return'},
      {id: '1.5.5', topic: 'Leadership commitment'},
    ],
    support: {tools: 0, methods: 0, recipes: 1, reference: 0, guidance: 0},
    resources: [],
    cookbook: [{to: `${CB}sustain`, label: 'Chapter 9, recipe 9.2: estimate the cost per query before launch'}],
    steps: null,
    evidence: [],
  },
  {
    id: '1.6',
    pillar: 1,
    name: 'Partnerships',
    short: 'Partnerships',
    coverage: 'partial',
    summary:
      'The program is a channel for partnership: organizations can adopt its open-source tools, contribute recipes and examples, and take part in its collaboration with peer organizations and the international statistical community (questions 1.6.2 and 1.6.4).',
    questions: [
      {id: '1.6.1', topic: 'Governance of external AI partnerships'},
      {id: '1.6.2', topic: 'Collaboration in the statistical community', covered: true},
      {id: '1.6.3', topic: 'National AI ecosystem coordination'},
      {id: '1.6.4', topic: 'Contribution to the broader AI community', covered: true},
    ],
    support: {tools: 1, methods: 0, recipes: 1, reference: 0, guidance: 2},
    resources: ['partnerships', 'openSource', 'cb'],
    cookbook: [{to: `${CB}contribute`, label: 'Contribute a recipe or an example from the organization'}],
    steps: [
      {from: 'A', to: 'B', text: 'Adopt one program tool and report back what worked.'},
      {from: 'B', to: 'C', text: 'Contribute an example or a recipe to the cookbook under the organization\'s name.'},
      {from: 'C', to: 'D', text: 'Co-develop methods and standards through the program partnerships and publish results as open source.'},
    ],
    evidence: ['Published contributions (recipes, code, examples) with the organization named'],
  },

  // --------------------------------------------------------------- Pillar II
  {
    id: '2.1',
    pillar: 2,
    name: 'Metadata standards and classifications',
    short: 'Metadata',
    coverage: 'direct',
    summary:
      'The program publishes the metadata schemas used by NADA and the Metadata Editor, develops methods to complete and check records with AI under curator review, and works on classifications, concept links, and a question bank. The cookbook translates these into a three-layer check and templates that an organization can run on its own catalog.',
    questions: [
      {id: '2.1.1', topic: 'Adoption of metadata standards', covered: true},
      {id: '2.1.2', topic: 'Classifications, concepts, vocabularies', covered: true},
      {id: '2.1.3', topic: 'Linkage between data and metadata', covered: true},
      {id: '2.1.4', topic: 'Machine-readable publication and web discoverability', covered: true},
      {id: '2.1.5', topic: 'Provenance and lineage', covered: true},
    ],
    support: {tools: 2, methods: 2, recipes: 2, reference: 2, guidance: 2},
    resources: ['metadataSchemas', 'metadataEditor', 'nada', 'metadataAugmentation', 'knowledgeGraphs', 'questionBank', 'standardsWs', 'microdataLibrary'],
    cookbook: [
      {to: `${CB}find`, label: 'Chapter 1: recipe 1.1 (check records against the schema and a profile), recipe 1.2 (schema.org on every page)'},
      {to: `${CB}understand`, label: 'Chapter 3: definitions and method notes, code lists and crosswalks, breaks and revisions as data'},
      {to: `${CB}standards`, label: 'Standards by chapter: SDMX, DDI, ISO 19115, Dublin Core, DCAT, DataCite'},
      {to: `${CB}data-types`, label: 'Application by data type: schema, format, and interface per type'},
    ],
    steps: [
      {from: 'A', to: 'B', text: 'Document each product in the World Bank schema for its type with the Metadata Editor or the CSV template, and run the completeness check at the foundational level.'},
      {from: 'B', to: 'C', text: 'Publish code lists and crosswalks, record breaks and provenance as fields, and emit schema.org or DCAT from the catalog (NADA does this on every page).'},
      {from: 'C', to: 'D', text: 'Reach the AI-ready profile across the catalog, link concepts to shared vocabularies, and use AI-assisted review under a curator to keep records consistent.'},
    ],
    evidence: ['Completeness report at the target level using the organization\'s profile', 'A validated schema.org or DCAT record', 'Published code lists and crosswalks', 'Provenance block in records'],
  },
  {
    id: '2.2',
    pillar: 2,
    name: 'Quality control procedures',
    short: 'Quality',
    coverage: 'direct',
    summary:
      'Anomaly detection with explained causes, metadata quality scoring, a curator review board, observation-level status codes, and numeric verification of generated answers address most questions in this dimension. The disclosure policy for generative AI (2.2.5) is covered by the cookbook\'s policy template and the responsible AI guidance.',
    questions: [
      {id: '2.2.1', topic: 'Quality assurance framework updated for AI', covered: true},
      {id: '2.2.2', topic: 'Data quality control procedures', covered: true},
      {id: '2.2.3', topic: 'Metadata quality process', covered: true},
      {id: '2.2.4', topic: 'Communicating data quality to users', covered: true},
      {id: '2.2.5', topic: 'Disclaimers for GenAI-assisted dissemination', covered: true},
      {id: '2.2.6', topic: 'Observation-level quality annotation', covered: true},
      {id: '2.2.7', topic: 'Methodological breaks and comparability', covered: true},
    ],
    support: {tools: 2, methods: 2, recipes: 2, reference: 1, guidance: 2},
    resources: ['anomaly', 'metadataQuality', 'metadataReviewer', 'pcn', 'responsibleAi', 'wdiSdmx'],
    cookbook: [
      {to: `${CB}understand`, label: 'Chapter 3, recipe 3.3: breaks and revisions as data with SDMX OBS_STATUS codes'},
      {to: `${CB}find`, label: 'Chapter 1, recipe 1.1: structure, completeness, and validity layers for metadata'},
      {to: `${CB}trust`, label: 'Chapter 5: source and release on every page; numbers verified before display'},
      {to: `${CB}govern`, label: 'Chapter 8: AI-use policy with disclosure rules and human review'},
    ],
    steps: [
      {from: 'A', to: 'B', text: 'Run the anomaly detector on the main series and the metadata check on the catalog; adopt OBS_STATUS codes in the values files.'},
      {from: 'B', to: 'C', text: 'Put the Metadata Reviewer or an equivalent review step in front of curators; publish a revisions log and a disclosure rule for AI-assisted content.'},
      {from: 'C', to: 'D', text: 'Explain flagged anomalies with cited evidence before release, verify every number in generated answers, and report quality measures with each release.'},
    ],
    evidence: ['Anomaly explanation logs with classifications and evidence', 'Metadata quality scores per record', 'OBS_STATUS in published files', 'Revisions log', 'Disclosure policy for generative AI'],
  },
  {
    id: '2.3',
    pillar: 2,
    name: 'Searchable data catalog',
    short: 'Catalog',
    coverage: 'direct',
    summary:
      'NADA provides the catalog, with schema.org markup, keyword and semantic search, and managed access; the Microdata Library runs on it with DataCite DOIs. The discoverability work and the PI-FT toolkit improve search by meaning over structured records. The cookbook provides the question sets and measures needed to demonstrate the improvement.',
    questions: [
      {id: '2.3.1', topic: 'Status of the data catalog', covered: true},
      {id: '2.3.2', topic: 'Search capabilities', covered: true},
      {id: '2.3.3', topic: 'Lineage and quality information in the catalog', covered: true},
      {id: '2.3.4', topic: 'Catalog currency and synchronization'},
      {id: '2.3.5', topic: 'Release calendar and change log'},
      {id: '2.3.6', topic: 'Persistent identifiers for datasets', covered: true},
    ],
    support: {tools: 2, methods: 2, recipes: 2, reference: 2, guidance: 1},
    resources: ['nada', 'microdataLibrary', 'discoverability', 'pift', 'metadataEditor'],
    cookbook: [
      {to: `${CB}find`, label: 'Chapter 1, recipe 1.3: semantic search next to keyword search, scored against a baseline'},
      {to: `${CB}trust`, label: 'Chapter 5, recipe 5.1: DOIs, citations, and licences on every page'},
      {to: `${CB}evaluate`, label: 'Chapter 6: known-item question sets and retrieval scores per language'},
    ],
    steps: [
      {from: 'A', to: 'B', text: 'Catalog every published product in NADA or an equivalent, with stable identifiers and keyword search.'},
      {from: 'B', to: 'C', text: 'Register DOIs, add semantic search, and write the question set that measures whether users find the right series.'},
      {from: 'C', to: 'D', text: 'Fine-tune retrieval on the organization\'s own records, report Recall@k and MRR per language, and surface lineage and quality fields in results.'},
    ],
    evidence: ['Catalog URL with schema.org records', 'DOI registrations', 'Retrieval scores per language with the question set'],
  },
  {
    id: '2.4',
    pillar: 2,
    name: 'API access',
    short: 'APIs',
    coverage: 'direct',
    summary:
      'NADA, which the program supports, provides catalog and metadata APIs out of the box, and the WDI SDMX API and the Data360 API are reference implementations of data APIs. The cookbooks cover the rest of the dimension: an SDMX or OpenAPI-described interface that returns provenance with every response, bulk files, rate limits and quotas with the status codes that clients act on, monitoring of availability and latency behind a gateway, and versioning with a revisions log. Scaling the API to the organization\'s own load remains its engineering work.',
    questions: [
      {id: '2.4.1', topic: 'Extent of API access', covered: true},
      {id: '2.4.2', topic: 'Documentation and usability', covered: true},
      {id: '2.4.3', topic: 'Performance, reliability, and machine-friendly features', covered: true},
      {id: '2.4.4', topic: 'Machine-scale features and bulk access', covered: true},
      {id: '2.4.5', topic: 'Dataset versioning and reproducibility', covered: true},
    ],
    support: {tools: 1, methods: 0, recipes: 3, reference: 2, guidance: 1},
    resources: ['wdiSdmx', 'data360Api', 'nada', 'cbStandards'],
    cookbook: [
      {to: `${CB}retrieve`, label: 'Chapter 2: tidy files with SDMX concepts, an OpenAPI description, bulk download, terms of use'},
      {to: `${CB}trust`, label: 'Chapter 5, recipe 5.2: provenance and licence in every API response'},
      {to: `${CB}understand`, label: 'Chapter 3, recipe 3.3: revisions file and status codes for reproducibility'},
      {to: '/cookbook/serving-statistics-to-agents/access', label: 'Serving Statistics to AI Agents, chapter 5: tiers, quotas, rate limits with 429 and Retry-After, client registry and audit'},
      {to: '/cookbook/serving-statistics-to-agents/operate', label: 'Serving Statistics to AI Agents, chapter 9: stable endpoint behind a gateway, availability and latency monitoring, versioning and deprecation'},
      {to: '/cookbook/ml-ready-datasets/version', label: 'Publishing ML-Ready Datasets, chapter 8: versioned releases with manifests and a DOI per version'},
    ],
    steps: [
      {from: 'A', to: 'B', text: 'Publish tidy CSV with SDMX column names at stable URLs and a bulk file; run NADA\'s catalog and metadata API where the catalog is NADA.'},
      {from: 'B', to: 'C', text: 'Describe the API with OpenAPI (or run an SDMX web service); return unit, period, release, and source with every response; monitor uptime and response times; add rate limits with 429 and Retry-After.'},
      {from: 'C', to: 'D', text: 'Add pagination, filtering, versioned datasets with a revisions log and retrievable vintages, published service targets and terms; test with the direct-request and stability checks.'},
    ],
    evidence: ['OpenAPI or SDMX description at a stable URL', 'A response that carries provenance', 'Monitoring dashboard or report of availability and latency; rate-limit policy', 'Revisions log and version identifiers'],
  },
  {
    id: '2.5',
    pillar: 2,
    name: 'Agentic AI access and generative AI for dissemination',
    short: 'Agent access',
    coverage: 'direct',
    summary:
      'The program\'s work is concentrated in this dimension: MCP design for official statistics with two servers in operation (Data360 and NADA), grounded answers with every number verified (Proof-Carrying Numbers), evaluation of retrieval and answers, synthetic data for sharing, ML-ready datasets with Croissant records and dataset cards, and small and agentic models for statistical tasks. The cookbook sequences these so that a conversational interface follows findability, retrieval, meaning, and trust.',
    questions: [
      {id: '2.5.1', topic: 'Generative AI for dissemination', covered: true},
      {id: '2.5.2', topic: 'Model Context Protocol adoption', covered: true},
      {id: '2.5.3', topic: 'AI-optimized formats and ML-ready datasets', covered: true},
      {id: '2.5.4', topic: 'RAG architecture and grounding assurance', covered: true},
      {id: '2.5.5', topic: 'Synthetic data generation and publication', covered: true},
    ],
    support: {tools: 2, methods: 2, recipes: 2, reference: 2, guidance: 2},
    resources: ['mcpDocs', 'data360Mcp', 'nada', 'pcn', 'synthetic', 'smallAgentic', 'inclusive', 'dataSnapshots'],
    cookbook: [
      {to: `${CB}retrieve`, label: 'Chapter 2, recipe 2.3: wrap the API as an MCP server (runs on mcp 1.x and 2.x)'},
      {to: `${CB}ask`, label: 'Chapter 4: answer from retrieved records only; monthly review of answers'},
      {to: `${CB}trust`, label: 'Chapter 5, recipe 5.3: verify the numbers in a generated answer'},
      {to: `${CB}evaluate`, label: 'Chapter 6, recipe 6.3: numeric accuracy, citation validity, refusal accuracy'},
      {to: `${CB}find`, label: 'Chapter 1: Croissant records for datasets meant for model training'},
      {to: '/cookbook/ml-ready-datasets/', label: 'Publishing ML-Ready Datasets: selection, Croissant records, dataset cards, representativeness, splits, licence, versions (question 2.5.3)'},
    ],
    steps: [
      {from: 'A', to: 'B', text: 'Expose the catalog and data through MCP tools over the existing API; start with read-only tools and provenance in every response.'},
      {from: 'B', to: 'C', text: 'Build the answer flow on retrieval first, verify numbers before display, and keep the question set that scores answers.'},
      {from: 'C', to: 'D', text: 'Report grounding measures with each release, publish ML-ready formats where appropriate, and use synthetic data where confidential microdata cannot be shared.'},
    ],
    evidence: ['MCP server with its tool list', 'Number verification logs', 'Answer-level evaluation results', 'Synthetic data release documentation', 'Croissant records'],
  },
  {
    id: '2.6',
    pillar: 2,
    name: 'Data licensing',
    short: 'Licensing',
    coverage: 'partial',
    summary:
      'The World Bank\'s default licence for datasets (CC BY 4.0), the licence field in the metadata schemas, and the cookbook\'s recipes place a machine-readable licence in every record, page, and API response. The ML-ready datasets cookbook states, beside the licence, that training is permitted and how models attribute the data (2.6.4); the legal decision remains the organization\'s.',
    questions: [
      {id: '2.6.1', topic: 'Type of licence on public data', covered: true},
      {id: '2.6.2', topic: 'Licence associated with each dataset', covered: true},
      {id: '2.6.3', topic: 'Machine-readable licensing information', covered: true},
      {id: '2.6.4', topic: 'Explicit coverage of AI and ML use', covered: true},
    ],
    support: {tools: 1, methods: 0, recipes: 3, reference: 2, guidance: 2},
    resources: ['metadataSchemas', 'metadataEditor', 'cbStandards'],
    cookbook: [
      {to: `${CB}trust`, label: 'Chapter 5, recipe 5.1: licence in the record, on the page, and in the citation'},
      {to: `${CB}find`, label: 'Chapter 1, recipe 1.2: licence in the schema.org record'},
      {to: `${CB}retrieve`, label: 'Chapter 2: licence and terms of use in the API description'},
      {to: '/cookbook/ml-ready-datasets/license', label: 'Publishing ML-Ready Datasets, chapter 7: licence and terms for model training, machine-readable in the Croissant record (question 2.6.4)'},
    ],
    steps: [
      {from: 'A', to: 'B', text: 'Choose one open licence (CC BY 4.0 is the World Bank default) and state it on every dataset page.'},
      {from: 'B', to: 'C', text: 'Fill the licence field in each record and expose it in schema.org and the API response.'},
      {from: 'C', to: 'D', text: 'State in the licence text how AI training and derived products are treated, with legal advice.'},
    ],
    evidence: ['Licence field present in every record (completeness check at AI-ready level)', 'Licence in schema.org and API responses'],
  },
];

export const coverageLabels = {
  direct: 'Program components address most questions',
  partial: 'Program components address some questions',
  none: 'No program component; the organization\'s own decision',
};
