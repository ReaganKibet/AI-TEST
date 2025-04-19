import os
import sqlite3
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

class MemoryManager:
    """
    Manages short-term and long-term memory storage for the application.
    Uses SQLite for persistence and session context for temporary memory.
    """
    def __init__(self, db_path="memory.db"):
        """
        Initialize the MemoryManager with a specified database path.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.session_memory = {}  # Short-term memory
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create creations table for long-term memory
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS creations (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                prompt TEXT,
                enhanced_prompt TEXT,
                image_path TEXT,
                model_path TEXT,
                metadata TEXT
            )
            ''')
            
            # Create tags table for searchable tags
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                creation_id TEXT,
                tag TEXT,
                FOREIGN KEY (creation_id) REFERENCES creations (id)
            )
            ''')
            
            conn.commit()
            conn.close()
            logging.info("Memory database initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing database: {e}")
            raise
    
    def store_session_data(self, session_id: str, key: str, value: Any):
        """
        Store data in short-term (session) memory.
        
        Args:
            session_id (str): Unique identifier for the session
            key (str): Key to store the value under
            value (Any): Value to store
        """
        if session_id not in self.session_memory:
            self.session_memory[session_id] = {}
        self.session_memory[session_id][key] = value
        
    def get_session_data(self, session_id: str, key: str) -> Optional[Any]:
        """
        Retrieve data from short-term (session) memory.
        
        Args:
            session_id (str): Unique identifier for the session
            key (str): Key to retrieve the value for
            
        Returns:
            Optional[Any]: The stored value, or None if not found
        """
        if session_id in self.session_memory and key in self.session_memory[session_id]:
            return self.session_memory[session_id][key]
        return None
    
    def store_creation(self, creation_data: Dict[str, Any]) -> str:
        """
        Store a creation in long-term memory.
        
        Args:
            creation_data (Dict[str, Any]): Data about the creation
            
        Returns:
            str: ID of the stored creation
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            creation_id = creation_data.get('id')
            timestamp = creation_data.get('timestamp', datetime.now().isoformat())
            prompt = creation_data.get('prompt', '')
            enhanced_prompt = creation_data.get('enhanced_prompt', '')
            image_path = creation_data.get('image_path', '')
            model_path = creation_data.get('model_path', '')
            metadata = json.dumps(creation_data.get('metadata', {}))
            
            # Insert into creations table
            cursor.execute('''
            INSERT INTO creations (id, timestamp, prompt, enhanced_prompt, image_path, model_path, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (creation_id, timestamp, prompt, enhanced_prompt, image_path, model_path, metadata))
            
            # Extract and store tags
            tags = self._extract_tags(prompt + " " + enhanced_prompt)
            for tag in tags:
                cursor.execute('''
                INSERT INTO tags (creation_id, tag) VALUES (?, ?)
                ''', (creation_id, tag.lower()))
            
            conn.commit()
            conn.close()
            
            logging.info(f"Creation stored with ID: {creation_id}")
            return creation_id
            
        except Exception as e:
            logging.error(f"Error storing creation: {e}")
            raise
    
    def _extract_tags(self, text: str) -> List[str]:
        """
        Extract relevant tags from text.
        
        Args:
            text (str): Text to extract tags from
            
        Returns:
            List[str]: List of extracted tags
        """
        # Simple tag extraction - split by spaces and filter
        words = text.lower().replace(",", " ").replace(".", " ").split()
        # Filter out common words and keep only relevant tags
        stopwords = {"a", "an", "the", "in", "on", "at", "with", "and", "or", "but", "to", "of", "for"}
        tags = [word for word in words if len(word) > 2 and word not in stopwords]
        return list(set(tags))  # Remove duplicates
    
    def search_by_prompt(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for creations by prompt similarity.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of matching creations
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Return results as dictionaries
            cursor = conn.cursor()
            
            # Extract search terms
            search_terms = self._extract_tags(query)
            if not search_terms:
                return []
            
            # Build query with OR conditions for each tag
            placeholders = ', '.join(['?'] * len(search_terms))
            cursor.execute(f'''
            SELECT DISTINCT c.* 
            FROM creations c
            JOIN tags t ON c.id = t.creation_id
            WHERE t.tag IN ({placeholders})
            ORDER BY c.timestamp DESC
            LIMIT ?
            ''', search_terms + [limit])
            
            results = []
            for row in cursor.fetchall():
                data = dict(row)
                data['metadata'] = json.loads(data['metadata'])
                results.append(data)
            
            conn.close()
            return results
            
        except Exception as e:
            logging.error(f"Error searching by prompt: {e}")
            return []
    
    def get_recent_creations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent creations.
        
        Args:
            limit (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of recent creations
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM creations
            ORDER BY timestamp DESC
            LIMIT ?
            ''', (limit,))
            
            results = []
            for row in cursor.fetchall():
                data = dict(row)
                data['metadata'] = json.loads(data['metadata'])
                results.append(data)
            
            conn.close()
            return results
            
        except Exception as e:
            logging.error(f"Error getting recent creations: {e}")
            return []