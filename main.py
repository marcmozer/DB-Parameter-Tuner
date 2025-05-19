import psycopg2
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
import random
import sys

class WorkMemOptimizer:
    def __init__(self, db_params, memory_range=(4, 1024), episodes=50):
        """
        Initializes the optimizer for PostgreSQL work_mem parameter.
        
        Args:
            db_params (dict): Connection parameters for the PostgreSQL database
            memory_range (tuple): Range of work_mem values in MB (min, max)
            episodes (int): Number of training episodes
        """
        self.db_params = db_params
        self.memory_min, self.memory_max = memory_range
        self.episodes = episodes
        self.state_size = 1  # Current work_mem value
        self.action_size = 3  # Increase, Decrease, Keep
        
        # Initialize Q-table
        # Discretize the state space (work_mem) into 50 steps
        self.state_buckets = 50
        self.q_table = np.zeros((self.state_buckets, self.action_size))
        
        # Hyperparameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.01
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.95  # Discount factor
        
        # Storage for results
        self.history = {
            "work_mem": [],
            "query_time": [],
            "rewards": []
        }
        
        # Load test queries
        self.test_queries = self.load_test_queries()
        
    def load_test_queries(self):
        """Loads the test SQL queries"""
        return [
            # Top 10 rated movies
            "SELECT t.primaryTitle, r.averageRating FROM imdb_test.titles t JOIN imdb_test.title_ratings r ON t.tconst = r.tconst WHERE t.titleType = 'movie' ORDER BY r.averageRating DESC LIMIT 10;",
            # Number of movies per year
            "SELECT startYear, COUNT(*) FROM imdb_test.titles WHERE titleType = 'movie' AND startYear IS NOT NULL GROUP BY startYear ORDER BY startYear DESC LIMIT 20;",
            # Actors with the most movies
            "SELECT n.primaryName, COUNT(*) FROM imdb_test.names n JOIN imdb_test.title_principals p ON n.nconst = p.nconst WHERE p.category = 'actor' GROUP BY n.primaryName ORDER BY COUNT(*) DESC LIMIT 10;",
            # Average rating by genre
            "SELECT genre, AVG(r.averageRating) FROM (SELECT t.tconst, unnest(string_to_array(t.genres, ',')) AS genre FROM imdb_test.titles t WHERE t.titleType = 'movie') g JOIN imdb_test.title_ratings r ON g.tconst = r.tconst GROUP BY genre HAVING COUNT(*) > 5 ORDER BY AVG(r.averageRating) DESC LIMIT 10;",
            # Episodes of a specific series
            "SELECT e.seasonNumber, e.episodeNumber, t.primaryTitle FROM imdb_test.title_episode e JOIN imdb_test.titles t ON e.tconst = t.tconst WHERE e.parentTconst IN (SELECT tconst FROM imdb_test.titles WHERE primaryTitle LIKE '%Breaking Bad%' LIMIT 1) ORDER BY e.seasonNumber, e.episodeNumber LIMIT 20;",
            # Directors with best average ratings
            "SELECT n.primaryName, AVG(r.averageRating), COUNT(*) as movie_count FROM imdb_test.title_crew c JOIN imdb_test.names n ON n.nconst = ANY(string_to_array(c.directors, ',')) JOIN imdb_test.title_ratings r ON c.tconst = r.tconst GROUP BY n.primaryName HAVING COUNT(*) > 2 ORDER BY AVG(r.averageRating) DESC LIMIT 10;",
            # Movies of a specific actor
            "SELECT t.primaryTitle, t.startYear FROM imdb_test.titles t JOIN imdb_test.title_principals p ON t.tconst = p.tconst WHERE p.nconst IN (SELECT nconst FROM imdb_test.names WHERE primaryName LIKE 'Meryl%' LIMIT 1) AND t.titleType = 'movie' ORDER BY t.startYear DESC LIMIT 10;",
            # Series with most episodes
            "SELECT t.primaryTitle, COUNT(*) as episode_count FROM imdb_test.titles t JOIN imdb_test.title_episode e ON t.tconst = e.parentTconst WHERE t.titleType = 'tvSeries' GROUP BY t.primaryTitle HAVING COUNT(*) > 3 ORDER BY COUNT(*) DESC LIMIT 5;",
            # Movies between two years with minimum rating
            "SELECT t.primaryTitle, t.startYear, r.averageRating FROM imdb_test.titles t JOIN imdb_test.title_ratings r ON t.tconst = r.tconst WHERE t.titleType = 'movie' AND t.startYear BETWEEN 2000 AND 2010 AND r.averageRating > 7.0 ORDER BY r.averageRating DESC LIMIT 15;",
            # Most frequent actor pairings
            "SELECT n1.primaryName AS actor1, n2.primaryName AS actor2, COUNT(*) FROM imdb_test.title_principals p1 JOIN imdb_test.title_principals p2 ON p1.tconst = p2.tconst AND p1.nconst < p2.nconst JOIN imdb_test.names n1 ON p1.nconst = n1.nconst JOIN imdb_test.names n2 ON p2.nconst = n2.nconst WHERE p1.category = 'actor' AND p2.category = 'actor' GROUP BY n1.primaryName, n2.primaryName HAVING COUNT(*) > 1 ORDER BY COUNT(*) DESC LIMIT 5;"
        ]
        
    def state_to_bucket(self, work_mem):
        """Converts a work_mem value into a bucket index for the Q-table"""
        log_min = np.log(self.memory_min)
        log_max = np.log(self.memory_max)
        log_mem = np.log(work_mem)
        
        normalized = (log_mem - log_min) / (log_max - log_min)
        bucket = int(normalized * (self.state_buckets - 1))
        return max(0, min(bucket, self.state_buckets - 1))
    
    def get_postgresql_work_mem(self):
        """
        Retrieves and prints the current work_mem value from a PostgreSQL database.
        """
        try:
            # Establish a connection to the PostgreSQL database
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()

            # Execute the query to get the current work_mem value
            cursor.execute("SHOW work_mem;")

            # Fetch the result
            work_mem_value = cursor.fetchone()[0]

            # Close the cursor and connection
            cursor.close()
            conn.close()

        except psycopg2.Error as e:
            print(f"Error connecting to or querying the database: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            sys.exit(1)
        return work_mem_value

    def set_work_mem(self, value_mb):
        """Sets the work_mem parameter in the database"""
        value_mb = max(self.memory_min, min(self.memory_max, value_mb))
        try:
            conn = psycopg2.connect(**self.db_params)
            conn.autocommit = True 
            try:
                with conn.cursor() as cur:
                    cur.execute(f"ALTER SYSTEM SET work_mem = '{value_mb}MB';")
                    cur.execute("SELECT pg_reload_conf();")
               
                
            finally:
                conn.close()
        except Exception as e:
            print(e) 
        return value_mb
    
    def execute_test_queries(self):
        """Executes test queries and measures the total execution time"""
        conn = psycopg2.connect(**self.db_params)
        total_time = 0
        
        try:
            with conn.cursor() as cur:
                # Pre-warm with EXPLAIN ANALYZE
                for query in self.test_queries:
                    try:
                        print(f"Preparing: {query}")
                        cur.execute(f"EXPLAIN ANALYZE {query}")
                        cur.fetchall()
                    except Exception as e:
                        print(f"Error preparing query: {e}")
                
                # Actual timing
                for query in self.test_queries:
                    try:
                        start_time = time.time()
                        cur.execute(query)
                        cur.fetchall()
                        query_time = time.time() - start_time
                        total_time += query_time
                    except Exception as e:
                        print(f"Error executing query: {e}")
                        total_time += 10  # Penalty for failed queries
        finally:
            conn.close()
            
        return total_time
    
    def get_action(self, state_bucket):
        """Selects an action based on the current policy"""
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.q_table[state_bucket])
    
    def apply_action(self, current_mem, action):
        """Applies the chosen action to the current work_mem value"""
        if action == 0:  # Increase
            step = current_mem * 0.1
            new_mem = current_mem + step
        elif action == 1:  # Decrease
            step = current_mem * 0.1
            new_mem = current_mem - step
        else:  # Slight variation
            step = current_mem * 0.05
            new_mem = current_mem + random.uniform(-step, step)
            
        return max(self.memory_min, min(self.memory_max, new_mem))
    
    def calculate_reward(self, query_time):
        """Reward is a negative of query time, so minimizing time maximizes reward"""
        return -query_time
    
    def test_with_default(self):
        """ Executes the query set with the default work_mem value"""
        default_mem = 4
        self.set_work_mem(default_mem)
        print(f"Current work_mem: {default_mem:.2f} MB")
        query_time = self.execute_test_queries()
        print(f"Total query time with default: {query_time:.4f} seconds")
    
    def train(self):
        """Trains the RL agent"""
        current_mem = np.sqrt(self.memory_min * self.memory_max)
        print("Episodes: ", self.episodes)
        for episode in range(self.episodes):
            
            

            print(f"\nEpisode {episode+1}/{self.episodes}")
            
            current_mem = self.set_work_mem(current_mem)
            actual_mem = self.get_postgresql_work_mem()
            print(f"Set work_mem to: {current_mem:.2f} MB")
            print(f"Currently set work_mem = {float(actual_mem.lower().replace('kb', '')) / 1024}") #test if value is set in database (first converts output to float)

            
            current_state = self.state_to_bucket(current_mem)
            query_time = self.execute_test_queries()
            print(f"Total query time: {query_time:.4f} seconds")
            
            reward = self.calculate_reward(query_time)
            print(f"Reward: {reward:.4f}")
            
            action = self.get_action(current_state)
            new_mem = self.apply_action(current_mem, action)
            new_state = self.state_to_bucket(new_mem)
            
            best_next_action = np.argmax(self.q_table[new_state])
            td_target = reward + self.gamma * self.q_table[new_state][best_next_action]
            td_error = td_target - self.q_table[current_state][action]
            self.q_table[current_state][action] += self.alpha * td_error
            
            current_mem = new_mem
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
            self.history["work_mem"].append(current_mem)
            self.history["query_time"].append(query_time)
            self.history["rewards"].append(reward)
            
    def find_optimal_setting(self):
        """Finds the optimal work_mem setting from training results"""
        best_episode = np.argmax(self.history["rewards"])
        optimal_mem = self.history["work_mem"][best_episode]
        optimal_time = self.history["query_time"][best_episode]
        
        print(f"\nOptimal work_mem setting: {optimal_mem:.2f} MB")
        print(f"Query time at optimal setting: {optimal_time:.4f} seconds")
        
        return optimal_mem
    
    def plot_results(self):
        """Visualizes training results"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
        
        ax1.plot(self.history["work_mem"])
        ax1.set_title('work_mem Values During Training')
        ax1.set_ylabel('work_mem (MB)')
        ax1.set_xlabel('Episode')
        
        ax2.plot(self.history["query_time"])
        ax2.set_title('Query Times During Training')
        ax2.set_ylabel('Time (Seconds)')
        ax2.set_xlabel('Episode')
        
        ax3.plot(self.history["rewards"])
        ax3.set_title('Rewards During Training')
        ax3.set_ylabel('Reward')
        ax3.set_xlabel('Episode')
        
        plt.tight_layout()
        plt.savefig('workmem_optimization_results.png')
        plt.show()

# Example usage
if __name__ == "__main__":
    log_file = open("training_log.txt", "w")
    sys.stdout = log_file
    
    db_params = {
        "host": "localhost",
        "database": "imdb_performance_tuning",
        "user": "postgres",
        "password": "MM2001"
    }
    
    optimizer = WorkMemOptimizer(
        db_params=db_params,
        memory_range=(4, 512),  # 4MB to 512MB
        episodes=50
    )
    
    optimizer.test_with_default()
    optimizer.train()
    optimal_mem = optimizer.find_optimal_setting()
    optimizer.plot_results()
    
    print(f"\nOptimization complete. Recommended work_mem setting: {optimal_mem:.2f} MB")

    # Restore stdout and close the log file
    sys.stdout = sys.__stdout__
    log_file.close()
