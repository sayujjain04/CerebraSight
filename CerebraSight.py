from openai import OpenAI
import os
from openai import AzureOpenAI
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
import string
import random

searcher_system_prompt_en = """## Character Introduction
You are an intelligent assistant that can call web search tools. Please collect information and reply to the question based on the current problem. You can use the following tools:
{tool_info}
## Reply Format

When calling the tool, please follow the format below:
```
Your thought process...<|action_start|><|plugin|>{{"name": "tool_name", "parameters": {{"param1": "value1"}}}}<|action_end|>
```

## Requirements

- Each key point in the response should be marked with the source of the search results to ensure the credibility of the information. The citation format is `[[int]]`. If there are multiple citations, use multiple [[]] to provide the index, such as `[[id_1]][[id_2]]`.
- Based on the search results of the "current problem", write a detailed and complete reply to answer the "current problem".
"""
FINAL_RESPONSE_EN = """Based on the provided Q&A pairs, write a detailed and comprehensive final response.
- The response content should be logically clear and well-structured to ensure reader understanding.
- Each key point in the response should be marked with the source of the search results (consistent with the indices in the Q&A pairs) to ensure information credibility. The index is in the form of `[[int]]`, and if there are multiple indices, use multiple `[[]]`, such as `[[id_1]][[id_2]]`.
- The response should be comprehensive and complete, without vague expressions like "based on the above content". The final response should not include the Q&A pairs provided to you.
- The language style should be professional and rigorous, avoiding colloquial expressions.
- Maintain consistent grammar and vocabulary usage to ensure overall document consistency and coherence."""

# Defines Client
client = AzureOpenAI(
  azure_endpoint = 'ENDPOINT', 
  api_key="API KEY",  
  api_version="2024-02-01" 
)
#Defines LLM
class LLM:
    def __init__(self, openai_client):
        self.openai_client = openai_client

    def call_gpt4(self, message):
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=message
        )
        return response.choices[0].message.content

#Defines Websearch Agent
class WebPlanner:
    def __init__(self, llm):
        self.graph = WebSearchGraph()
        self.searcher = WebSearcher(llm)
        self.llm = llm

    def add_root_node(self, question):
        self.graph.add_root_node(node_content=question, node_name="root")
        message = [
        {"role": "system", "content": searcher_system_prompt_en.format(tool_info="Web Search")},
        {"role": "user", "content": f"How would you break down the following query into sub-questions: {question}"}
        ]
        sub_questions = self.llm.call_gpt4(message)  # Call the LLM to generate sub-questions

    # Filter and refine the generated sub-questions
        clean_sub_questions = []
        for line in sub_questions.split('\n'):
            cleaned_line = line.strip().lstrip('1234567890.').strip()  # Remove leading numbers or bullets
            if cleaned_line and not cleaned_line.startswith(('-', '*', '1.', '2.', '3.', '4.', '5.')):
                clean_sub_questions.append(cleaned_line)

    # Add each clean sub-question to the graph and search
        for i, sub_question in enumerate(clean_sub_questions):
            if sub_question:  # Only add non-empty sub-questions
                self.add_sub_question_node(f"sub_question_{i+1}", sub_question)
    
        print(f"Added root node with question: {question}")

    def add_sub_question_node(self, node_name, sub_question):
        search_results = self.searcher.search_web(sub_question)
        if search_results:
            parsed_results = self.searcher.parse_search_results(search_results)
            top_result = parsed_results[0] if parsed_results else None
            if top_result and top_result['link']:  # Check if a valid link was found
                page_content = self.searcher.fetch_and_parse_page(top_result['link'])
                if page_content:
                    source_info = {
                    'number': len(self.graph.nodes) + 1,  # Source number
                    'name': top_result['title'],
                    'link': top_result['link']
                    }
                    summary = self.searcher.summarize_content(page_content, source_info)
                    self.graph.add_node(node_name=node_name, node_content=summary)
                    self.graph.add_edge(start_node="root", end_node=node_name)
                    print(f"Added sub-question node: {node_name} with content: {summary}")
                else:
                    print("No content found to summarize.")
            else:
                print("No search results found.")
        else:
            print("Failed to retrieve search results for the sub-question.")
    def finalize_response(self):
    # Gather all nodes' content
        nodes_content = [self.graph.node(node_name)['content'] for node_name in self.graph.nodes]
    
    # Create a message to generate the final response
        message = [
            {"role": "system", "content": FINAL_RESPONSE_EN},
            {"role": "user", "content": f"Based on the following information, provide a final comprehensive response: {nodes_content}"}
        ]
    
    # Use the LLM to generate the final answer
        final_answer = self.llm.call_gpt4(message)
        print(f"Final response: {final_answer}")
        return final_answer


#Defines WebPlanner Agent

class WebSearcher:
    def __init__(self, llm):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.llm = llm

    def sanitize_query(self, query: str) -> str:
        # Remove punctuation from the query
        return query.translate(str.maketrans('', '', string.punctuation))

    def get_top_google_search_links(self, query: str, num_results: int = 3, max_retries: int = 3):
        # Sanitize the query to remove any punctuation
        query = self.sanitize_query(query)

        # Construct the Google search URL
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        
        retries = 0
        while retries < max_retries:
            # Fetch the Google search results page
            response = requests.get(search_url, headers=self.headers)

            # Check if the request was successful
            if response.status_code != 200:
                print(f"Attempt {retries + 1} failed: {response.status_code}")
                retries += 1
                time.sleep(random.uniform(1, 5))  # Wait before retrying
                continue

            # Parse the content with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the search result links
            results = soup.select('div.yuRUbf a')  # Updated selector to match Google's structure

            # Extract the top search result links
            links = []
            for link in results:
                href = link['href']
                if href.startswith('http'):
                    links.append(href)
                if len(links) >= num_results:
                    break

            if links:
                return links
            
            print(f"Attempt {retries + 1} found no links.")
            retries += 1
            time.sleep(random.uniform(1, 5))  # Wait before retrying

        raise Exception("Failed to fetch search results after multiple retries.")

    def get_website_text(self, url: str) -> str:
        # Fetch the webpage
        response = requests.get(url, headers=self.headers)

        # Check if the request was successful
        if response.status_code != 200:
            print(f"Failed to fetch the webpage: {response.status_code}")
            return ""

        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the text content
        texts = soup.stripped_strings
        page_text = ' '.join(texts)

        # Mimic human behavior by adding a random sleep
        time.sleep(random.uniform(1, 3))

        return page_text

    def search_and_summarize(self, query: str):
        try:
            # Step 1: Get top Google search result links
            links = self.get_top_google_search_links(query, num_results=3)

            # Step 2: Fetch text content from those links
            combined_content = ""
            for i, link in enumerate(links, 1):
                print(f"Fetching content from link {i}: {link}")
                page_text = self.get_website_text(link)
                if page_text:
                    combined_content += page_text + "\n"
                else:
                    print(f"Failed to retrieve content from link {i}: {link}")

            if combined_content:
                # Step 3: Summarize the content using an LLM
                summary = self.summarize_content(combined_content, {'number': 1, 'name': 'Google', 'link': links[0]})
                return summary
            else:
                print("No valid content retrieved from any links.")
                return None

        except Exception as e:
            print(f"An error occurred during search and summarization: {e}")
            return None

    def summarize_content(self, content, source_info):
        message = [
            {"role": "system", "content": searcher_system_prompt_en.format(tool_info="Web Search")},
            {"role": "user", "content": f"Summarize the following content: {content}"}
        ]
        summary = self.llm.call_gpt4(message)
        return f"{summary} [{source_info['number']}, {source_info['name']}, {source_info['link']}]"

class WebPlanner:
    def __init__(self, llm):
        self.graph = WebSearchGraph()
        self.searcher = WebSearcher(llm)
        self.llm = llm

    def add_root_node(self, question):
        self.graph.add_root_node(node_content=question, node_name="root")
        message = [
            {"role": "system", "content": searcher_system_prompt_en.format(tool_info="Web Search")},
            {"role": "user", "content": f"How would you break down the following query into sub-questions: {question}"}
        ]
        sub_questions = self.llm.call_gpt4(message)

        # Filter the generated sub-questions
        clean_sub_questions = []
        for line in sub_questions.split('\n'):
            cleaned_line = line.strip().lstrip('1234567890.').strip()  # Remove leading numbers or bullets
            if cleaned_line and not cleaned_line.startswith(('-', '*', '1.', '2.', '3.', '4.', '5.')):
                clean_sub_questions.append(cleaned_line)

        # Add each clean sub-question to the graph and search
        for i, sub_question in enumerate(clean_sub_questions):
            if sub_question:  # Only add non-empty sub-questions
                self.add_sub_question_node(f"sub_question_{i+1}", sub_question)
        
        print(f"Added root node with question: {question}")

    def add_sub_question_node(self, node_name, sub_question):
        summary = self.searcher.search_and_summarize(sub_question)
        if summary:
            self.graph.add_node(node_name=node_name, node_content=summary)
            self.graph.add_edge(start_node="root", end_node=node_name)
            print(f"Added sub-question node: {node_name} with content: {summary}")
        else:
            print(f"No summary available for sub-question: {sub_question}")

    def finalize_response(self):
        nodes_content = [self.graph.node(node_name)['content'] for node_name in self.graph.nodes]
        message = [
            {"role": "system", "content": FINAL_RESPONSE_EN},
            {"role": "user", "content": f"Based on the following information, provide a final comprehensive response: {nodes_content}"}
        ]
        final_answer = self.llm.call_gpt4(message)
        print(f"Final response: {final_answer}")
        return final_answer


def main(query):
    llm = LLM(openai_client)
    planner = WebPlanner(llm)
    planner.add_root_node(query)
    final_response = planner.finalize_response()
    return final_response