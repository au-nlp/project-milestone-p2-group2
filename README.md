### Human-Like Next-Link Prediction - Group 2
### TODO
Readme.md file containing the detailed project proposal (up to 1000 words). Your README.md should contain:
Title
Abstract: A 150 word description of the project idea and goals. What’s the motivation behind your project? What story would you like to tell, and why?
Contributions: What's the contribution / novelty that you're aiming to produce.
Proposed additional datasets (if any): List the additional dataset(s) you want to use (if any), and some ideas on how you expect to get, manage, process, and enrich it/them. Show us that you’ve read the docs and some examples, and that you have a clear idea on what to expect. Discuss data size and format if relevant. It is your responsibility to check that what you propose is feasible.
Methods
Proposed timeline
Organization within the team: A list of internal milestones up until project Milestone P3.
Appendix section (not counted towards the 1000 words)
Repo organisation
Questions for TAs (optional): Add here any questions you have for us related to the proposed project.
### Deadline 23:59 CET, Fri 07 Nov 2025
### NOTE: P2 Milestone was created using GenAI tools. We used the help to create the helper functions as well as main.ipynb file.
### Issues that we faced: Data format - differences made it challenging to parse correctly in data loader.


# Structure in datafolder
- add folder "wikispeedia_articles_html" to store the html files
- "wikispeedia_articles_html" containts folder "wpcd"
- folder "wpcd" contains folder "css", "images", "js", "wp"

# Methods
---
**Information we have access to (similar to what the human would have access to):**
- current article
- goal article
- available links
- link positions
- path so far
- **to predict**
    - next link

**How to predict:**
1. **Baseline**
    - calculate for every candidate-link a score: $score(candidate) = cos(emb(candidate), emb(goal))$
    - choose candidate with the highest score
    - people choose articles that are semantically closer to the goal
2. **Heuristic (Taking hub bias into account)**
    - at the beginning, players are looking for hubs (articles with a lot of outgoing links (eg. Science, History, World))
    - we model this by incorporating a hub-feature
	    - $score(candidate) = a * cos_sim(candidate, goal) + (1-a) * (outdegree(candidate) / max(outdegree))$
			- a depends on the current status of the game
				- early in game: more weight on outdegree
				- later in game: more weight on semantic similarity
3. **Using training data from paths_finished**
	- first: train model with information a human has access to
	- second: incorporate historical data to learn how much each factor weights (eg. how much does the similarity between current article and goal article count?)
	- so we use historical data to weight the features of our heuristic

### TODO
Clean repo code, prepare readme, submit