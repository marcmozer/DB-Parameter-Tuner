CREATE TABLE titles (
    tconst VARCHAR(20) PRIMARY KEY,
    titleType VARCHAR(50),
    primaryTitle VARCHAR(500),
    originalTitle VARCHAR(500),
    isAdult BOOLEAN,
    startYear INTEGER,
    endYear INTEGER,
    runtimeMinutes INTEGER,
    genres VARCHAR(200)
);

CREATE TABLE names (
    nconst VARCHAR(20) PRIMARY KEY,
    primaryName VARCHAR(200),
    birthYear INTEGER,
    deathYear INTEGER,
    primaryProfession VARCHAR(200),
    knownForTitles VARCHAR(200)
);

CREATE TABLE title_crew (
    tconst VARCHAR(20) PRIMARY KEY,
    directors TEXT,
    writers TEXT
);

CREATE TABLE title_episode (
    tconst VARCHAR(20) PRIMARY KEY,
    parentTconst VARCHAR(20),
    seasonNumber INTEGER,
    episodeNumber INTEGER
);

CREATE TABLE title_ratings (
    tconst VARCHAR(20) PRIMARY KEY,
    averageRating NUMERIC(3,1),
    numVotes INTEGER
);

CREATE TABLE title_principals (
    tconst VARCHAR(20),
    ordering INTEGER,
    nconst VARCHAR(20),
    category VARCHAR(100),
    job TEXT,
    characters TEXT,
    PRIMARY KEY (tconst, ordering)
);

CREATE TABLE title_akas (
    titleId VARCHAR(20),
    ordering INTEGER,
    title TEXT,
    region VARCHAR(10),
    language VARCHAR(10),
    types VARCHAR(50),
    attributes TEXT,
    isOriginalTitle BOOLEAN,
    PRIMARY KEY (titleId, ordering)
);
