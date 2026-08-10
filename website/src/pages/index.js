import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import styles from './index.module.css';

const Stats = [
  {value: '1,400+', label: 'WDI Indicators'},
  {value: '217', label: 'Economies Covered'},
  {value: '6', label: 'Flagship Workstreams'},
  {value: 'Open', label: 'Source Tooling'},
];

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className={clsx('container', styles.heroInner)}>
        <span className={styles.badge}>
          <span className={styles.badgeDot} />
          World Bank &middot; Development Data Group
        </span>
        <Heading as="h1" className={styles.heroTitle}>
          AI for Data <span className={styles.accent}>—</span> Data for{' '}
          <span className={styles.accent}>AI</span>
        </Heading>
        <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className={clsx('button button--lg', styles.primaryButton)}
            to="/docs/introduction">
            Read the Documentation
          </Link>
          <Link
            className={clsx('button button--lg', styles.secondaryButton)}
            to="https://github.com/worldbank/ai4data">
            View on GitHub
          </Link>
        </div>
        <div className={styles.stats}>
          {Stats.map((stat) => (
            <div key={stat.label}>
              <div className={styles.statValue}>{stat.value}</div>
              <div className={styles.statLabel}>{stat.label}</div>
            </div>
          ))}
        </div>
      </div>
    </header>
  );
}

function MissionSection() {
  return (
    <section className={styles.missionSection}>
      <div className="container">
        <div className={styles.missionGrid}>
          <div className={styles.missionCard}>
            <span className={styles.missionEyebrow}>Mission 01</span>
            <Heading as="h3" className={styles.missionTitle}>
              AI for Data
            </Heading>
            <p className={styles.missionText}>
              Applying AI to improve data and metadata quality, data
              discoverability and dissemination, monitoring of data use, and
              user experience in producing and accessing development
              datasets.
            </p>
          </div>
          <div className={styles.missionCard}>
            <span className={styles.missionEyebrow}>Mission 02</span>
            <Heading as="h3" className={styles.missionTitle}>
              Data for AI
            </Heading>
            <p className={styles.missionText}>
              Ensuring development data is structured, documented, and made
              available in ways that enable effective and trustworthy use by
              AI systems.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="A World Bank Development Data Group program applying AI to improve development data, and making development data AI-ready.">
      <HomepageHeader />
      <main>
        <MissionSection />
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
