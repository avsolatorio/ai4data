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
        <span className="eyebrow">AI-readiness assessment · in practice</span>
        <Heading as="h1" className={styles.heroTitle}>
          From assessment to action
        </Heading>
        <p className={styles.heroLede}>
          The AI-readiness assessment tells a statistical office where it
          stands on twelve dimensions. This page shows, dimension by
          dimension, which program workstreams, open-source tools, reference
          implementations, and cookbook recipes an office can use to move up a
          level, and what each produces as evidence for the assessment form.
        </p>
        <div className={styles.heroActions}>
          <Link className={clsx('button button--md', styles.primary)} to="/ai-readiness-assessment">
            Open the assessment framework
          </Link>
          <Link className={clsx('button button--md', styles.secondary)} to="/cookbook/ai-ready-dissemination/start-here">
            Five-minute self-assessment
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
      title: 'Assess',
      body: 'Score each dimension with the framework, or take the cookbook\'s five-minute version to place a team quickly.',
      to: '/ai-readiness-assessment',
      cta: 'Assessment framework',
    },
    {
      n: '2',
      title: 'Map',
      body: 'Find the dimensions with the largest gaps below. Each one lists what the program provides and what the office has to decide itself.',
      to: '#explorer',
      cta: 'Dimension explorer',
    },
    {
      n: '3',
      title: 'Act',
      body: 'Follow the cookbook recipes and tools named for that dimension. Each produces an artifact the assessment form asks for as evidence.',
      to: '/cookbook/ai-ready-dissemination/',
      cta: 'Cookbook',
    },
  ];
  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.head}>
          <span className="eyebrow">How this page works</span>
          <Heading as="h2" className={styles.title}>
            Three steps, with the program in the third
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
          The program is strongest on Pillar II, the readiness of data
          products. On Pillar I it contributes to specific questions (open
          source, skills, interoperability standards, privacy-preserving
          techniques, evaluation, partnerships) and has no component for
          strategy, legal frameworks, infrastructure, or funding. The matrix
          below shows this honestly.
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
          <span className="eyebrow">Coverage at a glance</span>
          <Heading as="h2" className={styles.title}>
            What the program offers for each dimension
          </Heading>
          <p className={styles.lede}>
            Each cell shows how much of a kind of support exists for a
            dimension. Select a row to open it in the explorer.
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
          <span className="eyebrow">Dimension by dimension</span>
          <Heading as="h2" className={styles.title}>
            What to use, and what it produces
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
                  What the program offers
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
                  Cookbook path
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
                  Moving up a level with program resources
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
                  Levels A to D are the framework's maturity scale. The
                  framework's own suggested actions for each step apply as
                  well; these are the ones the program's resources support.
                </p>
              </div>
            ) : (
              <div className={styles.block}>
                <Heading as="h4" className={styles.blockTitle}>
                  Moving up a level
                </Heading>
                <p className={styles.fine}>
                  The program has no component for this dimension. The
                  framework's suggested actions and the office's own planning
                  apply.
                </p>
              </div>
            )}

            {d.evidence.length > 0 && (
              <div className={styles.block}>
                <Heading as="h4" className={styles.blockTitle}>
                  Evidence these produce for the assessment form
                </Heading>
                <ul className={styles.evidence}>
                  {d.evidence.map((e) => (
                    <li key={e}>{e}</li>
                  ))}
                </ul>
              </div>
            )}

            <footer className={styles.panelFoot}>
              <Link to={`/ai-readiness-assessment`}>Read the full dimension in the framework →</Link>
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
              Start with the gaps that cost the most
            </Heading>
            <p className={styles.closingText}>
              The framework's early-stage pathway puts quality control,
              metadata standards, a searchable catalog, and API access under a
              clear licence first. Those are dimensions 2.2, 2.1, 2.3, 2.4, and
              2.6 on this page, and the cookbook's chapters 1 to 3 and 5. A
              chatbot or agent interface (2.5) comes after.
            </p>
          </div>
          <div className={styles.closingActions}>
            <Link className={clsx('button button--md', styles.primary)} to="/cookbook/ai-ready-dissemination/">
              Open the cookbook
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
      title="From assessment to action"
      description="How the AI for Data – Data for AI program's tools, methods, and cookbook recipes address each dimension of the AI-readiness assessment framework.">
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
