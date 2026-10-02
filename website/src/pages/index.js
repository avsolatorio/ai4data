import clsx from "clsx";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import Heading from "@theme/Heading";
import ThemedImage from "@theme/ThemedImage";
import useBaseUrl from "@docusaurus/useBaseUrl";
import Challenge from "@site/src/components/Challenge";
import Program from "@site/src/components/Program";
import ScrollDemo from "@site/src/components/ScrollDemo";
import HeroVisual from "@site/src/components/HeroVisual";
import CookbookCta from "@site/src/components/CookbookCta";
import GetInvolved from "@site/src/components/GetInvolved";

import styles from "./index.module.css";

const DDG_URL = "https://www.worldbank.org/en/about/unit/unit-dec/dev";

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  const logoLight = useBaseUrl("/img/ddg-logo.png");
  const logoDark = useBaseUrl("/img/ddg-logo-dark.png");
  return (
    <header className={styles.heroBanner}>
      <div className="container">
        <div className={styles.heroGrid}>
          <div className={styles.heroInner}>
            <Link className={styles.heroLogo} to={DDG_URL}>
              <ThemedImage
                alt="World Bank Group, Development Data Group"
                sources={{ light: logoLight, dark: logoDark }}
              />
            </Link>
            <Link className={clsx("eyebrow", styles.eyebrowLink)} to={DDG_URL}>
              Office of the WBG Chief Statistician &amp; Development Data Group
            </Link>
            <Heading as="h1" className={styles.heroTitle}>
              Making development data <strong>AI-ready.</strong>
            </Heading>
            <p className={styles.heroSubtitle}>
              {siteConfig.tagline}
            </p>
            <div className={styles.buttons}>
              <Link
                className={clsx("button button--lg", styles.primaryButton)}
                to="/docs/introduction"
              >
                Read the Documentation
              </Link>
              <Link
                className={clsx("button button--lg", styles.secondaryButton)}
                to="https://github.com/worldbank/ai4data"
              >
                View on GitHub
              </Link>
            </div>
          </div>
          <div className={styles.heroVisual}>
            <HeroVisual />
          </div>
        </div>
      </div>
    </header>
  );
}

function ClosingCta() {
  return (
    <section className={styles.closing}>
      <div className="container">
        <div className={styles.closingRow}>
          <div className={styles.closingInner}>
            <Heading as="h2" className={styles.closingTitle}>
              Open-source tooling and documentation
            </Heading>
            <p className={styles.closingText}>
              Methods, software, and guidance are developed as open resources,
              with documentation for each released workstream.
            </p>
          </div>
          <Link
            className={clsx("button button--md", styles.closingButtonPrimary)}
            to="https://github.com/worldbank/ai4data"
          >
            Explore the Repository
          </Link>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description={siteConfig.tagline}
    >
      <HomepageHeader />
      <main>
        <Challenge />
        <ScrollDemo />
        <Program />
        <CookbookCta />
        <GetInvolved />
        <ClosingCta />
      </main>
    </Layout>
  );
}
