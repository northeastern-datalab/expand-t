This repository contains code for `Expand-T: Demonstrating Table Reclamation and Expansion`.

At submission time, we have a Jupyter Notebook (`sample_demo.ipynb`) that runs Expand-T end-to-end. Upon acceptance, we will create a web application for users to thoroughly explore Expand-T. 

Expand-T consists of three steps: Table Reclamation, Table Expansion, and Table Exploration. For Table Reclamation, we adopt code from [Gen-T: Table Reclamation in Data Lakes](https://arxiv.org/abs/2403.14128), which is publicly available [here](https://github.com/northeastern-datalab/gen-t). 

Given a Source Table and a data lake, we first discover relevant tables (termed candidate tables) from the data lake and prune this set to only include tables needed to fully reclaim the Source Table (termed originating tables). This code can be found in the `discovery/` folder:
1.  We first retrieve an initial set of relevant tables from the data lake, using an existing table discovery method. By default, we use Starmie (https://github.com/megagonlabs/starmie). 

2. In `discovery/discover_candidates.py`, we get a set of Candidate Tables either from the set of tables returned from step (1) or from the data lake. To do so, we find tables containing columns with high set overlap with columns in the Source Table.

3. Gen-T now prunes the set of candidate tables to a set of originating tables, found in (`discovery/prune_candidates.py`). Here, Gen-T only keeps tables that are needed to reproduce the Source Table when integrated.

With a set of originating tables, we now integrate them with the goal of reproducing the Source Table (Table Reclamation) and expanding the Source Table (Table Expansion) We also include operations that users can use to explore the expanded Source Table (Table Exploration). This code can be found in the `integration/` folder:
1. In `targeted_integration.py`, we use an adaptation of Gen-T to integrate the set of originating tables. Specifically to integrate the tables in order to reclaim the Source Table, we run `integrate_tables()`. 
2. In the same file, we also expand the source table using `expand_tables()`.
3. Lastly, also in the same file, we can rank tuples (`rank_tuples()`) by a specified column and filter tuples (`filter_tuples()`) by column values.
4. To see the result of directly applying Outer Join on the set of candidate tables (thus replacing Gen-T and Table Expansion), we run `integration_utils.outerjoin()`.