import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import {cookbooks} from '@site/src/content/cookbooks';
import styles from './styles.module.css';

export default function CookbookLanding() {
  return (
    <Layout
      title="Cookbooks"
      description="Practical, problem-oriented guides from the AI for Data – Data for AI program.">
      <header className={styles.hero}>
        <div className="container">
          <span className="eyebrow">Cookbooks</span>
          <Heading as="h1" className={styles.title}>
            Practical guides for AI-ready data
          </Heading>
          <p className={styles.lede}>
            Each cookbook is organized around the questions a data team faces,
            with steps at different maturity levels, open standards, tests, and
            checklists. Select a cookbook to open it.
          </p>
        </div>
      </header>
      <main className={styles.main}>
        <div className="container">
          <div className={styles.grid}>
            {cookbooks.map((c) => (
              <Link className={styles.card} to={c.to} key={c.id}>
                <span className={styles.audience}>{c.audience}</span>
                <Heading as="h2" className={styles.cardTitle}>
                  {c.title}
                </Heading>
                <p className={styles.cardBody}>{c.description}</p>
                <span className={styles.meta}>
                  {c.chapters} chapters <span aria-hidden="true">→</span>
                </span>
              </Link>
            ))}
          </div>
        </div>
      </main>
    </Layout>
  );
}
