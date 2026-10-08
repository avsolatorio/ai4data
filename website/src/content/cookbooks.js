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
];
