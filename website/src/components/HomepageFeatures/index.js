import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const WorkstreamList = [
  {
    title: 'Generative AI for Metadata Quality',
    to: '/docs/metadata-quality/generative-ai-for-metadata-quality',
    description:
      'LLMs assess and improve metadata completeness, consistency, and semantic alignment across indicator catalogs.',
  },
  {
    title: 'Metadata Augmentation',
    to: '/docs/metadata-augmentation/',
    description:
      'Automated thematic tagging and enrichment of microdata data dictionaries using semantic clustering and LLMs.',
  },
  {
    title: 'Anomaly Detection and Explanation',
    to: '/docs/anomaly-detection/',
    description:
      'Statistical detection combined with LLM elicitation to classify and explain unusual patterns in timeseries data.',
  },
  {
    title: 'Data Discoverability',
    to: '/docs/data-discoverability/',
    description:
      'Semantic search systems enabling natural language queries over development datasets.',
  },
  {
    title: 'Model Context Protocol',
    to: '/docs/mcp/',
    description:
      'Enabling AI assistants to query official statistics directly via an open standard.',
  },
  {
    title: 'Inclusive AI Applications',
    to: '/docs/inclusive-ai/',
    description:
      'Approaches to extend AI benefits to low-resource contexts and languages.',
  },
];

function Workstream({title, to, description}) {
  return (
    <div className={clsx('col col--4')}>
      <Link to={to} className={styles.card}>
        <Heading as="h3" className={styles.cardTitle}>
          {title}
        </Heading>
        <p className={styles.cardDescription}>{description}</p>
        <span className={styles.cardLink}>Read more →</span>
      </Link>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <Heading as="h2" className={styles.sectionTitle}>
          Flagship Workstreams
        </Heading>
        <div className="row">
          {WorkstreamList.map((props, idx) => (
            <Workstream key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
