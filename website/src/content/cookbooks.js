// Cookbooks listed on the /cookbook landing page. Each cookbook is a folder
// under /cookbook with its own sidebar in sidebars-cookbook.js.
export const cookbooks = [
  {
    id: 'ai-ready-dissemination',
    audience: 'National statistical organizations',
    title: 'Practical Guide to AI-Ready Data Dissemination',
    description:
      'Nine questions a statistical organization can ask about its dissemination system. Each chapter has recipes with code and templates, steps at three maturity levels, tests, and a checklist. Includes a self-assessment.',
    chapters: 9,
    to: '/cookbook/ai-ready-dissemination/',
  },
  {
    id: 'microdata-documentation',
    audience: 'Data curators in national statistical organizations',
    title: 'Practical Guide to AI-Ready Microdata Documentation',
    description:
      'Eight questions a curator can ask about a survey\'s documentation, from producing it out of existing files to comparability across rounds and access conditions, each answered with recipes, maturity levels, tests, and a checklist. Builds on DDI Codebook and the World Bank microdata schema.',
    chapters: 8,
    to: '/cookbook/microdata-documentation/',
  },
  {
    id: 'data-from-documents',
    audience: 'Dissemination, library, and research data teams',
    title: 'Practical Guide to Extracting Data from Documents',
    description:
      'Nine questions a team can ask about the statistics locked in its reports and PDFs, each answered with recipes for reading scans, locating, extracting, verifying, combining, publishing, and citing the data, and running the pipeline at scale. Builds on the Data Snapshots work.',
    chapters: 9,
    to: '/cookbook/data-from-documents/',
  },
  {
    id: 'monitoring-data-use',
    audience: 'Dissemination teams and management in national statistical organizations',
    title: 'Practical Guide to Monitoring Data Use',
    description:
      'Nine questions a dissemination team can ask about how its data are used: counting access and citations, collecting research, policy, and web documents, finding and matching mentions, reporting use, and acting on it. Builds on the program\'s Monitoring of Data Use work.',
    chapters: 9,
    to: '/cookbook/monitoring-data-use/',
  },
  {
    id: 'serving-statistics-to-agents',
    audience: 'Developers and dissemination teams in national statistical organizations',
    title: 'Practical Guide to Serving Official Statistics to AI Agents',
    description:
      'Nine questions an organization can ask before and after exposing its statistics to AI agents: scope, tools, context, provenance, access control, safety, evaluation, distribution, and operation, each answered with recipes and checks. Builds on the Data360 MCP server and the program\'s MCP work.',
    chapters: 9,
    to: '/cookbook/serving-statistics-to-agents/',
  },
  {
    id: 'metadata-curation-with-llms',
    audience: 'Metadata curators and their managers in national statistical organizations',
    title: 'Practical Guide to Metadata Curation with Language Models',
    description:
      'Eight questions a curation team can ask about using language models for metadata: scope, migration of legacy records, drafting, assessment, review, vocabularies, translation, and improvement from curator decisions, each answered with recipes and scripts. Builds on the program\'s metadata quality and Metadata Reviewer work.',
    chapters: 8,
    to: '/cookbook/metadata-curation-with-llms/',
  },
  {
    id: 'evaluation-suites',
    audience: 'Teams that deploy AI components in national statistical organizations',
    title: 'Practical Guide to Evaluation Suites for Statistical AI',
    description:
      'How to build, run, and maintain the evaluation suites that decide whether search, assistants, extraction, and curation tools are good enough to publish and safe to change.',
    chapters: 9,
    to: '/cookbook/evaluation-suites/',
  },
];
