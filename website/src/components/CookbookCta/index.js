import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const Chapters = [
  {to: '/cookbook/ai-ready-dissemination/find', n: '1', q: 'Can people and AI find our statistics?'},
  {to: '/cookbook/ai-ready-dissemination/retrieve', n: '2', q: 'Can AI retrieve our data reliably?'},
  {to: '/cookbook/ai-ready-dissemination/understand', n: '3', q: 'Can AI understand what the numbers mean?'},
  {to: '/cookbook/ai-ready-dissemination/ask', n: '4', q: 'Can users ask questions naturally?'},
  {to: '/cookbook/ai-ready-dissemination/trust', n: '5', q: 'Can answers be trusted and traced?'},
  {to: '/cookbook/ai-ready-dissemination/evaluate', n: '6', q: 'Can we tell whether it works?'},
  {to: '/cookbook/ai-ready-dissemination/monitor-use', n: '7', q: 'Can we see how our data are used?'},
  {to: '/cookbook/ai-ready-dissemination/govern', n: '8', q: 'Can we operate this responsibly?'},
  {to: '/cookbook/ai-ready-dissemination/sustain', n: '9', q: 'Can we maintain it?'},
];

export default function CookbookCta() {
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.layout}>
          <div className={styles.text}>
            <span className="eyebrow">For statistical offices</span>
            <Heading as="h2" className={styles.title}>
              Practical Guide to AI-Ready Data Dissemination
            </Heading>
            <p className={styles.lede}>
              A cookbook organized around nine questions a national statistical
              office can ask about its own dissemination system. Each chapter
              lists steps at three maturity levels (foundational, AI-ready, and
              AI-native), implementation options based on open standards,
              tests, and a checklist. Generative AI is optional; the first
              steps are complete metadata, open data, stable identifiers, and
              documented APIs.
            </p>
            <Link
              className={clsx('button button--md', styles.button)}
              to="/cookbook/ai-ready-dissemination/">
              Open the guide
            </Link>
          </div>
          <ol className={styles.list}>
            {Chapters.map((c) => (
              <li key={c.to}>
                <Link className={styles.item} to={c.to}>
                  <span className={styles.num}>{c.n}</span>
                  <span>{c.q}</span>
                </Link>
              </li>
            ))}
          </ol>
        </div>
      </div>
    </section>
  );
}
