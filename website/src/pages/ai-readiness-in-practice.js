import {Fragment, useEffect, useMemo, useState} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import {
  FRAMEWORK_VERSION,
  coverageLabels,
  dimensions,
  resources,
  supportKinds,
} from '@site/src/content/readinessMap';
import styles from './ai-readiness-in-practice.module.css';

const PILLARS = {
  1: {label: 'Pillar I', name: 'Institutional readiness for AI adoption', tag: 'AI for Data'},
  2: {label: 'Pillar II', name: 'Readiness of data products for AI', tag: 'Data for AI'},
};

function hashToId() {
  if (typeof window === 'undefined') {
    return null;
  }
  const m = window.location.hash.match(/^#dim-(\d\.\d)$/);
  return m ? m[1] : null;
}

function Hero({counts}) {
  return (
    <header className={styles.hero}>
      <div className="container">
        <span className="eyebrow">AI-readiness assessment framework · operational companion</span>
        <Heading as="h1" className={styles.heroTitle}>
          Operationalizing the AI-readiness assessment
        </Heading>
        <p className={styles.heroLede}>
          The AI-readiness assessment framework scores a national statistical
          organization on twelve dimensions across two pillars. This page sets
          out, for each dimension, the program workstreams, open-source tools,
          reference implementations, and cookbook recipes that support
          progress to a higher maturity level. It also identifies the evidence
          that each resource produces for the assessment form.
        </p>
        <div className={styles.heroActions}>
          <Link className={clsx('button button--md', styles.primary)} to="/ai-readiness-assessment">
            Assessment framework
          </Link>
          <Link className={clsx('button button--md', styles.secondary)} to="/cookbook/ai-ready-dissemination/start-here">
            Cookbook self-assessment
          </Link>
          <Link className={clsx('button button--md', styles.secondary)} to="/modern-nso">
            A modern statistical office
          </Link>
        </div>
        <dl className={styles.facts}>
          <div>
            <dt>Dimensions</dt>
            <dd>12</dd>
          </div>
          <div>
            <dt>Dimension questions</dt>
            <dd>{counts.questions}</dd>
          </div>
          <div>
            <dt>Addressed by a program component</dt>
            <dd>{counts.covered}</dd>
          </div>
          <div>
            <dt>Cookbook chapters</dt>
            <dd>9</dd>
          </div>
        </dl>
      </div>
    </header>
  );
}

function HowItWorks() {
  const steps = [
    {
      n: '1',
      title: 'Assessment',
      body: 'Complete the assessment framework to obtain a maturity level (A to D) for each dimension. The cookbook provides a short self-assessment for a first orientation.',
      to: '/ai-readiness-assessment',
      cta: 'Assessment framework',
    },
    {
      n: '2',
      title: 'Mapping',
      body: 'For each dimension where the gap between the current and target levels is largest, consult its entry below. The entry lists the program resources that apply and the questions that remain the organization\'s own.',
      to: '#explorer',
      cta: 'Dimension entries',
    },
    {
      n: '3',
      title: 'Implementation',
      body: 'Apply the cookbook recipes and tools identified for the dimension. Each produces a document, dataset, or measurement that can be attached to the assessment form as evidence.',
      to: '/cookbook/ai-ready-dissemination/',
      cta: 'Cookbook',
    },
  ];
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">Purpose and structure</span>
          <Heading as="h2" className={styles.title}>
            Using the assessment results
          </Heading>
        </div>
        <ol className={styles.steps}>
          {steps.map((s) => (
            <li key={s.n} className={styles.step}>
              <span className={styles.stepNum}>{s.n}</span>
              <Heading as="h3" className={styles.stepTitle}>
                {s.title}
              </Heading>
              <p>{s.body}</p>
              {s.to.startsWith('#') ? (
                <a href={s.to}>{s.cta} →</a>
              ) : (
                <Link to={s.to}>{s.cta} →</Link>
              )}
            </li>
          ))}
        </ol>
        <p className={styles.note}>
          The program's contribution is concentrated in Pillar II, the
          readiness of data products. In Pillar I it addresses specific
          questions on open-source publication, skills, interoperability
          standards, privacy-preserving techniques, evaluation, and
          partnerships. It has no component for strategy, legal frameworks,
          infrastructure, or funding. The matrix below records this.
        </p>
      </div>
    </section>
  );
}

function Matrix({onPick}) {
  return (
    <section className={clsx(styles.section, styles.band)}>
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">Coverage by dimension</span>
          <Heading as="h2" className={styles.title}>
            Program support by assessment dimension
          </Heading>
          <p className={styles.lede}>
            Each cell indicates the extent of one kind of support for a
            dimension. Selecting a dimension opens its entry below.
          </p>
        </div>
        <div className={styles.matrixWrap}>
          <table className={styles.matrix}>
            <thead>
              <tr>
                <th scope="col">Dimension</th>
                {supportKinds.map((k) => (
                  <th scope="col" key={k.id}>
                    <span className={styles.kindFull}>{k.label}</span>
                    <span className={styles.kindShort}>{k.short}</span>
                  </th>
                ))}
                <th scope="col">Coverage</th>
              </tr>
            </thead>
            <tbody>
              {[1, 2].map((p) => (
                <Fragment key={p}>
                  <tr className={styles.matrixGroup}>
                    <th scope="rowgroup" colSpan={supportKinds.length + 2}>
                      {PILLARS[p].label} · {PILLARS[p].name}
                    </th>
                  </tr>
                  {dimensions
                    .filter((d) => d.pillar === p)
                    .map((d) => (
                      <tr key={d.id}>
                        <th scope="row">
                          <button type="button" className={styles.rowButton} onClick={() => onPick(d.id)}>
                            <span className={styles.dimId}>{d.id}</span> {d.name}
                          </button>
                        </th>
                        {supportKinds.map((k) => {
                          const v = d.support[k.id];
                          return (
                            <td key={k.id} className={styles.cell}>
                              <span
                                className={clsx(styles.dot, v === 2 && styles.dotFull, v === 1 && styles.dotHalf)}
                                aria-label={v === 2 ? 'substantial' : v === 1 ? 'some' : 'none'}
                                title={`${k.label}: ${v === 2 ? 'substantial' : v === 1 ? 'some' : 'none'}`}
                              />
                            </td>
                          );
                        })}
                        <td>
                          <span className={clsx(styles.badge, styles[`cov_${d.coverage}`])}>{d.coverage}</span>
                        </td>
                      </tr>
                    ))}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
        <p className={styles.legend}>
          <span className={clsx(styles.dot, styles.dotFull)} /> substantial&nbsp;&nbsp;
          <span className={clsx(styles.dot, styles.dotHalf)} /> some&nbsp;&nbsp;
          <span className={styles.dot} /> none
        </p>
      </div>
    </section>
  );
}

function Explorer({selected, setSelected}) {
  const d = dimensions.find((x) => x.id === selected);
  const pillar = PILLARS[d.pillar];
  const byKind = useMemo(() => {
    const groups = {};
    d.resources.forEach((key) => {
      const r = resources[key];
      (groups[r.kind] = groups[r.kind] || []).push(r);
    });
    return groups;
  }, [d]);
  const covered = d.questions.filter((q) => q.covered).length;

  return (
    <section className={styles.section} id="explorer">
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">Dimension entries</span>
          <Heading as="h2" className={styles.title}>
            Program resources and evidence for each dimension
          </Heading>
        </div>
        <div className={styles.explorer}>
          <nav className={styles.rail} aria-label="Dimensions">
            {[1, 2].map((p) => (
              <div key={p} className={styles.railGroup}>
                <span className={styles.railLabel}>
                  {PILLARS[p].label} · {PILLARS[p].tag}
                </span>
                {dimensions
                  .filter((x) => x.pillar === p)
                  .map((x) => (
                    <button
                      type="button"
                      key={x.id}
                      className={clsx(styles.railItem, x.id === selected && styles.railItemOn)}
                      aria-current={x.id === selected ? 'true' : undefined}
                      onClick={() => setSelected(x.id)}>
                      <span className={styles.dimId}>{x.id}</span>
                      <span>{x.short}</span>
                      <span className={clsx(styles.miniDot, styles[`cov_${x.coverage}`])} aria-hidden="true" />
                    </button>
                  ))}
              </div>
            ))}
          </nav>

          <article className={styles.panel} key={d.id}>
            <header className={styles.panelHead}>
              <span className={styles.panelKicker}>
                {pillar.label} · {pillar.name}
              </span>
              <Heading as="h3" className={styles.panelTitle}>
                <span className={styles.dimIdBig}>{d.id}</span> {d.name}
              </Heading>
              <span className={clsx(styles.badge, styles[`cov_${d.coverage}`])}>{coverageLabels[d.coverage]}</span>
            </header>

            <p className={styles.panelSummary}>{d.summary}</p>

            <div className={styles.block}>
              <Heading as="h4" className={styles.blockTitle}>
                Questions in this dimension{' '}
                <span className={styles.blockMeta}>
                  {covered} of {d.questions.length} addressed by a program component
                </span>
              </Heading>
              <ul className={styles.questions}>
                {d.questions.map((q) => (
                  <li key={q.id} className={clsx(styles.question, q.covered && styles.questionOn)}>
                    <span className={styles.qId}>{q.id}</span>
                    <span>{q.topic}</span>
                  </li>
                ))}
              </ul>
            </div>

            {d.resources.length > 0 && (
              <div className={styles.block}>
                <Heading as="h4" className={styles.blockTitle}>
                  Program resources
                </Heading>
                <div className={styles.kinds}>
                  {supportKinds
                    .filter((k) => byKind[k.id])
                    .map((k) => (
                      <div key={k.id} className={styles.kind}>
                        <span className={styles.kindLabel}>{k.label}</span>
                        <ul className={styles.resList}>
                          {byKind[k.id].map((r) => (
                            <li key={r.title}>
                              {r.to ? <Link to={r.to}>{r.title}</Link> : <strong>{r.title}</strong>}
                              <span className={styles.resNote}>{r.note}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ))}
                </div>
              </div>
            )}

            {d.cookbook.length > 0 && (
              <div className={styles.block}>
                <Heading as="h4" className={styles.blockTitle}>
                  Relevant cookbook chapters
                </Heading>
                <ul className={styles.cbList}>
                  {d.cookbook.map((c) => (
                    <li key={c.to + c.label}>
                      <Link to={c.to}>{c.label}</Link>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {d.steps ? (
              <div className={styles.block}>
                <Heading as="h4" className={styles.blockTitle}>
                  Progression between maturity levels
                </Heading>
                <ol className={styles.levels}>
                  {d.steps.map((s) => (
                    <li key={s.from} className={styles.level}>
                      <span className={styles.levelTag}>
                        {s.from} <span aria-hidden="true">→</span> {s.to}
                      </span>
                      <p>{s.text}</p>
                    </li>
                  ))}
                </ol>
                <p className={styles.fine}>
                  Levels A to D are the maturity scale of the framework. The
                  actions listed are those supported by program resources;
                  the framework's suggested actions for each transition also
                  apply.
                </p>
              </div>
            ) : (
              <div className={styles.block}>
                <Heading as="h4" className={styles.blockTitle}>
                  Progression between maturity levels
                </Heading>
                <p className={styles.fine}>
                  The program has no component for this dimension. The
                  framework's suggested actions and the organization's own
                  planning apply.
                </p>
              </div>
            )}

            {d.evidence.length > 0 && (
              <div className={styles.block}>
                <Heading as="h4" className={styles.blockTitle}>
                  Evidence for the assessment form
                </Heading>
                <ul className={styles.evidence}>
                  {d.evidence.map((e) => (
                    <li key={e}>{e}</li>
                  ))}
                </ul>
              </div>
            )}

            <footer className={styles.panelFoot}>
              <Link to={`/ai-readiness-assessment`}>Full dimension in the assessment framework →</Link>
              <span className={styles.permalink}>
                Link to this dimension: <code>#dim-{d.id}</code>
              </span>
            </footer>
          </article>
        </div>
      </div>
    </section>
  );
}

function Closing() {
  return (
    <section className={clsx(styles.section, styles.closing)}>
      <div className="container">
        <div className={styles.closingRow}>
          <div>
            <Heading as="h2" className={styles.closingTitle}>
              Recommended sequence
            </Heading>
            <p className={styles.closingText}>
              The framework's pathway for early-stage organizations places
              data quality control, metadata standards, a searchable catalog,
              and API access under a clear licence first. These correspond to
              dimensions 2.2, 2.1, 2.3, 2.4, and 2.6 on this page and to
              chapters 1 to 3 and 5 of the cookbook. Agentic and generative AI
              access (dimension 2.5) follows.
            </p>
          </div>
          <div className={styles.closingActions}>
            <Link className={clsx('button button--md', styles.primary)} to="/cookbook/ai-ready-dissemination/">
              Cookbook
            </Link>
            <Link className={clsx('button button--md', styles.secondaryOnDark)} to="mailto:ai4data@worldbank.org">
              Contact the program
            </Link>
          </div>
        </div>
        <p className={styles.version}>
          Question numbering follows the {FRAMEWORK_VERSION}. The mapping is
          maintained with the site; corrections are welcome as{' '}
          <Link to="https://github.com/worldbank/ai4data/issues">GitHub issues</Link>.
        </p>
      </div>
    </section>
  );
}


export default function InPractice() {
  const [selected, setSelected] = useState('2.1');
  const counts = useMemo(() => {
    const all = dimensions.flatMap((d) => d.questions);
    return {questions: all.length, covered: all.filter((q) => q.covered).length};
  }, []);

  useEffect(() => {
    const fromHash = hashToId();
    if (fromHash && dimensions.some((d) => d.id === fromHash)) {
      setSelected(fromHash);
      document.getElementById('explorer')?.scrollIntoView({block: 'start'});
    }
  }, []);

  const pick = (id) => {
    setSelected(id);
    if (typeof window !== 'undefined') {
      window.history.replaceState(null, '', `#dim-${id}`);
      document.getElementById('explorer')?.scrollIntoView({behavior: 'smooth', block: 'start'});
    }
  };

  return (
    <Layout
      title="Operationalizing the AI-readiness assessment"
      description="The program workstreams, tools, reference implementations, and cookbook recipes that support each dimension of the AI-readiness assessment framework, with the evidence each produces.">
      <Hero counts={counts} />
      <main>
        <HowItWorks />
        <Matrix onPick={pick} />
        <Explorer selected={selected} setSelected={pick} />
        <Closing />
      </main>
    </Layout>
  );
}
