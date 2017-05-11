Music Genre Classification of the Million Song Dataset
================================
Date:  05/10/2017
Contributors:  Alan Gee (ag64793), Ken Lee (kl22943) , Timothy Mahler (tam2643),
Farzan Memarian (fm6996), Andrew Stier (acs3235)
Repository URL: https://github.com/tmahler14/song_classifier

--------------------------------
Description
--------------------------------
- Contains project outline and description file & final report
- Contains all relevant code to construct models for predicting genres ('/src/models')
	- SEE final report detailing models constructed
- Contains all database manipulation and data parsing scripts ('/src/scripts')
- Contains all results of models ('/src/results')
- Contains all lyrics web scraper ('/src/lyrics')
- Contains all utility functions for accessing MySQL database and reading HDF5 files

--------------------------------
Installation
--------------------------------
- Run dump_sql.sql file into 'songs' database to replicate database
- Get lyrical dataset from AWS EBS (https://aws.amazon.com/datasets/million-song-dataset/)
- Install all python modules

--------------------------------
Directory Structure
--------------------------------
- src/
	- lyrics/
	- models/
	- scripts/
	- results/
	- util/