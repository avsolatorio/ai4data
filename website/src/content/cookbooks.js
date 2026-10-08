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
      'Six questions a curator can ask about a survey\'s documentation, each answered with recipes, maturity levels, tests, and a checklist. Builds on DDI Codebook and the World Bank microdata schema.',
    chapters: 6,
    to: '/cookbook/microdata-documentation/',
  },
  {
    id: 'data-from-documents',
    audience: 'Dissemination, library, and research data teams',
    title: 'Practical Guide to Extracting Data from Documents',
    description:
      'Six questions a team can ask about the statistics locked in its reports and PDFs, each answered with recipes for locating, extracting, verifying, publishing, and citing the data. Builds on the Data Snapshots work.',
    chapters: 6,
    to: '/cookbook/data-from-documents/',
  },
  {
    id: 'monitoring-data-use',
    audience: 'Dissemination teams and management in national statistical organizations',
    title: 'Practical Guide to Monitoring Data Use',
    description:
      'Six questions an organization can ask about where its data are used, each answered with recipes for defining use, collecting documents, detecting and harmonizing mentions, reporting, and acting on the results. Builds on the Monitoring of Data Use workstream.',
    chapters: 6,
    to: '/cookbook/monitoring-data-use/',
  },
  {
    id: 'serving-statistics-to-agents',
    audience: 'Developers and dissemination teams in national statistical organizations',
    title: 'Practical Guide to Serving Official Statistics to AI Agents',
    description:
      'Six questions an organization can ask before and after exposing its statistics to AI agents through the Model Context Protocol, each answered with recipes for tool design, provenance, safety, evaluation, and operation. Builds on the program\'s MCP work and the Data360 MCP server.',
    chapters: 6,
    to: '/cookbook/serving-statistics-to-agents/',
  },
  {
    id: 'metadata-curation-with-llms',
    audience: 'Metadata curators and their managers in national statistical organizations',
    title: 'Practical Guide to Metadata Curation with Language Models',
    description:
      'Six questions a curation team can ask before and after using language models to draft, assess, and standardize metadata, each answered with recipes that keep the curator in charge. Builds on the Generative AI for Metadata Quality work and the Metadata Reviewer.',
    chapters: 6,
    to: '/cookbook/metadata-curation-with-llms/',
  },
];
