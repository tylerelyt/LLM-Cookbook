#!/usr/bin/env python3
"""
NL2SQL Demo: Complete Text-to-SQL Pipeline
==========================================

A comprehensive demonstration of converting natural language questions to SQL queries
and providing intelligent answers based on database results.

Pipeline:
1. Vector retrieval of relevant DDL schemas
2. LLM-powered SQL generation using DDL context
3. SQLite database query execution
4. Intelligent answer generation from query results

Author: Tyler
License: MIT
"""

import os
import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from dashscope import Generation
import dashscope

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TableSchema:
    """Represents a database table schema with metadata."""
    name: str
    ddl: str
    description: str
    sample_queries: List[str]
    
@dataclass
class QueryResult:
    """Represents the result of a SQL query execution."""
    success: bool
    data: List[Dict[str, Any]]
    error: Optional[str] = None
    sql: Optional[str] = None

class NL2SQLPipeline:
    """
    Complete NL2SQL pipeline with vector retrieval and LLM generation.
    """
    
    def __init__(self, db_path: str = "demo.db", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the NL2SQL pipeline.
        
        Args:
            db_path: Path to SQLite database
            model_name: SentenceTransformer model for embeddings
        """
        self.db_path = db_path
        self.embedding_model = SentenceTransformer(model_name)
        self.schemas: List[TableSchema] = []
        self.schema_embeddings: Optional[np.ndarray] = None
        
        # Initialize LLM client
        self.llm_client = self._init_llm_client()
        
        # Initialize database and schemas
        self._init_database()
        self._load_schemas()
        self._build_schema_embeddings()
        
    def _init_llm_client(self) -> Optional[Any]:
        """Initialize LLM client (OpenAI or DashScope)."""
        dashscope_key = os.environ.get("DASHSCOPE_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        
        if dashscope_key:
            dashscope.api_key = dashscope_key
            logger.info("Using DashScope API")
            return "dashscope"
        elif openai_key:
            openai.api_key = openai_key
            logger.info("Using OpenAI API")
            return "openai"
        else:
            logger.warning("No API key found. Please set DASHSCOPE_API_KEY or OPENAI_API_KEY")
            return None
    
    def _init_database(self):
        """Initialize SQLite database with sample data."""
        logger.info(f"Initializing database: {self.db_path}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create employees table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            salary REAL NOT NULL,
            hire_date TEXT NOT NULL,
            manager_id INTEGER,
            FOREIGN KEY (manager_id) REFERENCES employees (id)
        )
        """)
        
        # Create departments table  
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            budget REAL NOT NULL,
            location TEXT NOT NULL
        )
        """)
        
        # Create projects table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department_id INTEGER NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT,
            budget REAL NOT NULL,
            status TEXT DEFAULT 'active',
            FOREIGN KEY (department_id) REFERENCES departments (id)
        )
        """)
        
        # Insert sample data if tables are empty
        cursor.execute("SELECT COUNT(*) FROM employees")
        if cursor.fetchone()[0] == 0:
            self._insert_sample_data(cursor)
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def _insert_sample_data(self, cursor):
        """Insert comprehensive sample data for testing."""
        
        # Insert departments
        departments = [
            (1, 'Engineering', 2000000, 'San Francisco'),
            (2, 'Marketing', 800000, 'New York'),
            (3, 'Sales', 1200000, 'Chicago'),
            (4, 'HR', 500000, 'Austin'),
            (5, 'Finance', 600000, 'Boston')
        ]
        cursor.executemany(
            "INSERT OR REPLACE INTO departments (id, name, budget, location) VALUES (?, ?, ?, ?)",
            departments
        )
        
        # Insert employees
        employees = [
            (1, 'Alice Johnson', 'Engineering', 120000, '2020-01-15', None),
            (2, 'Bob Smith', 'Engineering', 95000, '2021-03-22', 1),
            (3, 'Carol Davis', 'Engineering', 110000, '2019-07-30', 1),
            (4, 'David Wilson', 'Marketing', 75000, '2022-02-14', None),
            (5, 'Eve Brown', 'Marketing', 82000, '2021-11-08', 4),
            (6, 'Frank Miller', 'Sales', 65000, '2023-01-10', None),
            (7, 'Grace Lee', 'Sales', 70000, '2022-09-05', 6),
            (8, 'Henry Zhang', 'Engineering', 105000, '2020-12-01', 1),
            (9, 'Ivy Chen', 'HR', 60000, '2021-06-15', None),
            (10, 'Jack Taylor', 'Finance', 85000, '2020-08-20', None)
        ]
        cursor.executemany(
            "INSERT OR REPLACE INTO employees (id, name, department, salary, hire_date, manager_id) VALUES (?, ?, ?, ?, ?, ?)",
            employees
        )
        
        # Insert projects
        projects = [
            (1, 'AI Platform Development', 1, '2023-01-01', '2024-06-30', 500000, 'active'),
            (2, 'Mobile App Redesign', 1, '2023-03-15', '2023-12-31', 300000, 'completed'),
            (3, 'Brand Campaign 2023', 2, '2023-02-01', '2023-11-30', 150000, 'active'),
            (4, 'Sales Analytics Dashboard', 3, '2023-04-01', '2023-10-31', 200000, 'active'),
            (5, 'Employee Training Program', 4, '2023-01-15', '2023-12-15', 75000, 'active')
        ]
        cursor.executemany(
            "INSERT OR REPLACE INTO projects (id, name, department_id, start_date, end_date, budget, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            projects
        )
        
        logger.info("Sample data inserted successfully")
    
    def _load_schemas(self):
        """Load table schemas with descriptions and sample queries."""
        self.schemas = [
            TableSchema(
                name="employees",
                ddl="""
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    salary REAL NOT NULL,
    hire_date TEXT NOT NULL,
    manager_id INTEGER,
    FOREIGN KEY (manager_id) REFERENCES employees (id)
);
                """.strip(),
                description="Employee information including personal details, salary, and management hierarchy",
                sample_queries=[
                    "SELECT * FROM employees WHERE department = 'Engineering'",
                    "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 5",
                    "SELECT AVG(salary) FROM employees GROUP BY department"
                ]
            ),
            TableSchema(
                name="departments", 
                ddl="""
CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    budget REAL NOT NULL,
    location TEXT NOT NULL
);
                """.strip(),
                description="Department information including budget and location details",
                sample_queries=[
                    "SELECT * FROM departments ORDER BY budget DESC",
                    "SELECT name, location FROM departments",
                    "SELECT SUM(budget) FROM departments"
                ]
            ),
            TableSchema(
                name="projects",
                ddl="""
CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department_id INTEGER NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT,
    budget REAL NOT NULL,
    status TEXT DEFAULT 'active',
    FOREIGN KEY (department_id) REFERENCES departments (id)
);
                """.strip(),
                description="Project information including timeline, budget, and status tracking",
                sample_queries=[
                    "SELECT * FROM projects WHERE status = 'active'",
                    "SELECT p.name, d.name as department FROM projects p JOIN departments d ON p.department_id = d.id",
                    "SELECT SUM(budget) FROM projects WHERE status = 'active'"
                ]
            )
        ]
        logger.info(f"Loaded {len(self.schemas)} table schemas")
    
    def _build_schema_embeddings(self):
        """Build embeddings for schema descriptions and DDL."""
        if not self.schemas:
            return
            
        # Combine DDL and description for better semantic matching
        schema_texts = []
        for schema in self.schemas:
            text = f"Table: {schema.name}\nDescription: {schema.description}\nDDL: {schema.ddl}"
            schema_texts.append(text)
        
        self.schema_embeddings = self.embedding_model.encode(schema_texts)
        logger.info("Schema embeddings built successfully")
    
    def retrieve_relevant_schemas(self, question: str, top_k: int = 2) -> List[TableSchema]:
        """
        Retrieve relevant table schemas based on question similarity.
        
        Args:
            question: Natural language question
            top_k: Number of top schemas to retrieve
            
        Returns:
            List of relevant table schemas
        """
        if self.schema_embeddings is None:
            return self.schemas[:top_k]
        
        # Encode the question
        question_embedding = self.embedding_model.encode([question])
        
        # Calculate similarities
        similarities = cosine_similarity(question_embedding, self.schema_embeddings)[0]
        
        # Get top-k most similar schemas
        top_indices = np.argsort(similarities)[::-1][:top_k]
        relevant_schemas = [self.schemas[i] for i in top_indices]
        
        logger.info(f"Retrieved {len(relevant_schemas)} relevant schemas for question: {question}")
        for i, schema in enumerate(relevant_schemas):
            logger.info(f"  {i+1}. {schema.name} (similarity: {similarities[top_indices[i]]:.3f})")
            
        return relevant_schemas
    
    def generate_sql(self, question: str, schemas: List[TableSchema]) -> str:
        """
        Generate SQL query using LLM with relevant schemas.
        
        Args:
            question: Natural language question
            schemas: Relevant table schemas
            
        Returns:
            Generated SQL query
        """
        # Build context with relevant schemas
        schema_context = "\n\n".join([
            f"Table: {schema.name}\n{schema.ddl}\nDescription: {schema.description}"
            for schema in schemas
        ])
        
        prompt = f"""
You are an expert SQL developer. Given the database schema and a natural language question, generate a precise SQL query.

Database Schema:
{schema_context}

Question: {question}

Requirements:
1. Generate valid SQLite syntax
2. Use appropriate JOINs when needed
3. Include proper WHERE clauses for filtering
4. Use aggregate functions when appropriate
5. Return only the SQL query without explanation

SQL Query:"""

        try:
            if self.llm_client == "dashscope":
                response = Generation.call(
                    model="qwen-plus",
                    prompt=prompt,
                    result_format='message'
                )
                sql = response.output.choices[0].message.content.strip()
            elif self.llm_client == "openai":
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.1
                )
                sql = response.choices[0].message.content.strip()
            else:
                # Fallback: simple rule-based SQL generation
                sql = self._fallback_sql_generation(question, schemas)
            
            # Clean up the SQL (remove markdown formatting if present)
            sql = sql.replace("```sql", "").replace("```", "").strip()
            
            logger.info(f"Generated SQL: {sql}")
            return sql
            
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            return self._fallback_sql_generation(question, schemas)
    
    def _fallback_sql_generation(self, question: str, schemas: List[TableSchema]) -> str:
        """Simple fallback SQL generation based on keywords."""
        question_lower = question.lower()
        
        if "salary" in question_lower and "average" in question_lower:
            return "SELECT department, AVG(salary) as avg_salary FROM employees GROUP BY department"
        elif "employee" in question_lower and "engineering" in question_lower:
            return "SELECT * FROM employees WHERE department = 'Engineering'"
        elif "project" in question_lower and "active" in question_lower:
            return "SELECT * FROM projects WHERE status = 'active'"
        elif "department" in question_lower and "budget" in question_lower:
            return "SELECT name, budget FROM departments ORDER BY budget DESC"
        else:
            # Default query
            return f"SELECT * FROM {schemas[0].name} LIMIT 5"
    
    def execute_sql(self, sql: str) -> QueryResult:
        """
        Execute SQL query against the database.
        
        Args:
            sql: SQL query to execute
            
        Returns:
            Query execution result
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            cursor = conn.cursor()
            
            cursor.execute(sql)
            rows = cursor.fetchall()
            
            # Convert rows to list of dictionaries
            data = [dict(row) for row in rows]
            
            conn.close()
            
            logger.info(f"SQL executed successfully, returned {len(data)} rows")
            return QueryResult(success=True, data=data, sql=sql)
            
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            return QueryResult(success=False, data=[], error=str(e), sql=sql)
    
    def generate_answer(self, question: str, query_result: QueryResult) -> str:
        """
        Generate natural language answer from query results.
        
        Args:
            question: Original natural language question
            query_result: SQL query execution result
            
        Returns:
            Natural language answer
        """
        if not query_result.success:
            return f"I apologize, but I encountered an error while querying the database: {query_result.error}"
        
        if not query_result.data:
            return "I couldn't find any data matching your question in the database."
        
        # Format the data for the LLM
        data_summary = self._format_data_summary(query_result.data)
        
        prompt = f"""
Based on the database query results, provide a clear and concise answer to the user's question.

Original Question: {question}
SQL Query: {query_result.sql}

Query Results:
{data_summary}

Please provide a natural language answer that:
1. Directly addresses the user's question
2. Includes relevant details from the data
3. Is easy to understand
4. Mentions specific numbers/values when relevant

Answer:"""

        try:
            if self.llm_client == "dashscope":
                response = Generation.call(
                    model="qwen-plus",
                    prompt=prompt,
                    result_format='message'
                )
                answer = response.output.choices[0].message.content.strip()
            elif self.llm_client == "openai":
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.3
                )
                answer = response.choices[0].message.content.strip()
            else:
                # Fallback: simple answer generation
                answer = self._fallback_answer_generation(question, query_result.data)
            
            logger.info("Generated natural language answer")
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return self._fallback_answer_generation(question, query_result.data)
    
    def _format_data_summary(self, data: List[Dict[str, Any]]) -> str:
        """Format query results for LLM consumption."""
        if not data:
            return "No data found"
        
        # Limit data size for LLM context
        sample_data = data[:10]  # Show first 10 rows
        
        result_lines = []
        for i, row in enumerate(sample_data, 1):
            row_str = ", ".join([f"{k}: {v}" for k, v in row.items()])
            result_lines.append(f"{i}. {row_str}")
        
        if len(data) > 10:
            result_lines.append(f"... and {len(data) - 10} more rows")
        
        return "\n".join(result_lines)
    
    def _fallback_answer_generation(self, question: str, data: List[Dict[str, Any]]) -> str:
        """Simple fallback answer generation."""
        if not data:
            return "No results found for your question."
        
        count = len(data)
        
        if count == 1:
            row = data[0]
            details = ", ".join([f"{k}: {v}" for k, v in row.items()])
            return f"I found 1 result: {details}"
        else:
            return f"I found {count} results. Here are some key details: {self._format_data_summary(data[:3])}"
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Complete NL2SQL pipeline: question -> SQL -> answer.
        
        Args:
            question: Natural language question
            
        Returns:
            Complete pipeline result
        """
        logger.info(f"Processing question: {question}")
        
        # Step 1: Retrieve relevant schemas
        relevant_schemas = self.retrieve_relevant_schemas(question)
        
        # Step 2: Generate SQL
        sql = self.generate_sql(question, relevant_schemas)
        
        # Step 3: Execute SQL
        query_result = self.execute_sql(sql)
        
        # Step 4: Generate answer
        answer = self.generate_answer(question, query_result)
        
        return {
            "question": question,
            "relevant_schemas": [s.name for s in relevant_schemas],
            "sql": sql,
            "query_success": query_result.success,
            "query_error": query_result.error,
            "data_count": len(query_result.data) if query_result.success else 0,
            "answer": answer,
            "raw_data": query_result.data if query_result.success else None
        }

def demo_interactive_session():
    """Run an interactive demo session."""
    print("🤖 NL2SQL Interactive Demo")
    print("=" * 50)
    print("Ask questions about our company database!")
    print("Type 'quit' to exit\n")
    
    # Initialize pipeline
    try:
        pipeline = NL2SQLPipeline()
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        print("💡 Make sure to set DASHSCOPE_API_KEY or OPENAI_API_KEY environment variable")
        return
    
    # Sample questions for users
    sample_questions = [
        "What is the average salary by department?",
        "Who are the employees in the Engineering department?",
        "Which projects are currently active?",
        "What is the total budget of all departments?",
        "Show me the highest paid employees",
        "Which department has the largest budget?"
    ]
    
    print("💡 Sample questions you can try:")
    for i, q in enumerate(sample_questions, 1):
        print(f"   {i}. {q}")
    print()
    
    while True:
        try:
            question = input("🔍 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            print("\n⏳ Processing your question...")
            
            # Process the question
            result = pipeline.ask_question(question)
            
            # Display results
            print(f"\n📊 Results:")
            print(f"   🗂️  Relevant tables: {', '.join(result['relevant_schemas'])}")
            print(f"   🔍 Generated SQL: {result['sql']}")
            
            if result['query_success']:
                print(f"   ✅ Query executed successfully ({result['data_count']} rows)")
                print(f"\n💬 Answer:")
                print(f"   {result['answer']}")
            else:
                print(f"   ❌ Query failed: {result['query_error']}")
            
            print("\n" + "-" * 50 + "\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print()
    
    print("\n👋 Thanks for using NL2SQL Demo!")

def demo_batch_questions():
    """Run demo with predefined questions."""
    print("🤖 NL2SQL Batch Demo")
    print("=" * 50)
    
    # Initialize pipeline
    try:
        pipeline = NL2SQLPipeline()
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return
    
    # Test questions
    test_questions = [
        "What is the average salary by department?",
        "Who are the employees in the Engineering department?", 
        "Which projects are currently active?",
        "What is the total budget of all departments?",
        "Show me employees with salary greater than 100000"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📋 Question {i}: {question}")
        print("-" * 30)
        
        try:
            result = pipeline.ask_question(question)
            
            print(f"🗂️  Tables used: {', '.join(result['relevant_schemas'])}")
            print(f"🔍 SQL: {result['sql']}")
            
            if result['query_success']:
                print(f"✅ Found {result['data_count']} results")
                print(f"💬 Answer: {result['answer']}")
            else:
                print(f"❌ Error: {result['query_error']}")
                
        except Exception as e:
            print(f"❌ Failed to process question: {e}")

if __name__ == "__main__":
    import sys
    
    # Check for required packages
    try:
        import sentence_transformers
        import sklearn
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("💡 Install with: pip install sentence-transformers scikit-learn")
        sys.exit(1)
    
    # Run demo based on command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        demo_batch_questions()
    else:
        demo_interactive_session() 