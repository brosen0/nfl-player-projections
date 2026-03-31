name: llm-council
description: "Run any question,
idea, or decision through a
council of 5 AI advisors who
independently analyze it, peer-
review each other
anonymously, and synthesize a
final verdict. Based on
Karpathy's LLM Council
methodology. MANDATORY
TRIGGERS: 'council this', 'run
the council', 'war room this',
'pressure-test this', 'stress-test
this', 'debate this'. STRONG
TRIGGERS (use when
combined with a real decision
or tradeoff): 'should I X or Y',
'which option', 'what would you
do', 'is this the right move',
'validate this', 'get multiple
perspectives', 'I can't decide',
'I'm torn between'. Do NOT
trigger on simple yes/no
questions, factual lookups, or
casual 'should I' without a
meaningful tradeoff (e.g.
'should I use markdown' is not
a council question). DO trigger
when the user presents a
genuine decision with stakes,
multiple options, and context
that suggests they want it
pressure-tested from multiple
angles."
---
# LLM Council
https://docs.google.com/document/d/e/2PACX-1vSvw_Mk4iq4DkBC2TaEVBUDMjU4RtwDrKdxenpc-x7Vnzw5THGA4wVJd-LX/pub 3/30/26, 8:44 PM
Page 1 of 16
You ask one AI a question, you
get one answer. That answer
might be great. It might be mid.
You have no way to tell
because you only saw one
perspective.
The council fixes this. It runs
your question through 5
independent advisors, each
thinking from a fundamentally
different angle. Then they
review each other's work. Then
a chairman synthesizes
everything into a final
recommendation that tells you
where the advisors agree,
where they clash, and what you
should actually do.
This is adapted from Andrej
Karpathy's LLM Council. He
dispatches queries to multiple
models, has them peer-review
each other anonymously, then
a chairman produces the final
answer. We do the same thing
inside Claude using sub-agents
with different thinking lenses
instead of different models.
---
## when to run the council
The council is for questions
where being wrong is
expensive.
Good council questions:
- "Should I launch a $97
workshop or a $497 course?"
- "Which of these 3 positioning
angles is strongest?"
- "I'm thinking of pivoting from X
to Y. Am I crazy?"
- "Here's my landing page copy.
What's weak?"
- "Should I hire a VA or build an
automation first?"
https://docs.google.com/document/d/e/2PACX-1vSvw_Mk4iq4DkBC2TaEVBUDMjU4RtwDrKdxenpc-x7Vnzw5THGA4wVJd-LX/pub 3/30/26, 8:44 PM
Page 2 of 16
Bad council questions:
- "What's the capital of
France?" (one right answer, no
need for perspectives)
- "Write me a tweet" (creation
task, not a decision)
- "Summarize this article"
(processing task, not judgment)
The council shines when
there's genuine uncertainty and
the cost of a bad call is high. If
you already know the answer
and just want validation, the
council will likely tell you things
