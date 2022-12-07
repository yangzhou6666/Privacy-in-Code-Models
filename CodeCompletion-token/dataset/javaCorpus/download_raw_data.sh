# Obtain the data separation information
mkdir separation
cd separation
wget https://groups.inf.ed.ac.uk/cup/javaGithub/repositoryState.tar.gz
tar -xzvf repositoryState.tar.gz
cd ..

Obtain the raw data
mkdir JavaCorpusRaw
cd JavaCorpusRaw
wget https://groups.inf.ed.ac.uk/cup/javaGithub/java_projects.tar.gz
tar -xzvf java_projects.tar.gz