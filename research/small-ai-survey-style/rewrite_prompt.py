"""The two prompts of the survey-style run, copied from rewrite.py and rewrite_test.py."""
PROMPT = """A household survey asks two questions about work: "What is your main job?" and "What are your main tasks or duties in this job?". The interviewer writes the answer down in the respondent's own words.

A respondent holds this job:
Job title in a job advertisement: {title}
Occupation group: {label}

Write three different answers that this respondent could give.
- Answer 1: a short answer of one to four words, the way many respondents answer.
- Answer 2: the job together with the place of work or what the employer does, in four to twelve words.
- Answer 3: the main tasks in everyday words, in six to twenty words, without the job title.

Rules:
- Use the plain words of a respondent, and avoid the wording of a job advertisement. Do not copy the job title as it is written.
- Write the way an interviewer records an answer: mostly lower case, short, sometimes an incomplete sentence, in the first person or with no subject.
- Leave out the seniority grade, the company name, and anything a respondent would not say.
- Each answer must describe this job, and must fit this occupation group better than any related group."""

TEST_PROMPT = """In a labour force survey the interviewer asks "What kind of work do you do in your main job?" and writes down what the respondent says.

The respondent's job, as an employer would advertise it: {title}
Occupation group of this job: {name}

Write the answer this respondent would give, as spoken: between three and twenty words that describe the work. The answer must not repeat the advertised job title word for word."""
