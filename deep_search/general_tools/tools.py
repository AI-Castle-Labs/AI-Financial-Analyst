from llama_cloud_services import LlamaParse
from langchain.schema import Document
import re
import requests
import requests
from bs4 import BeautifulSoup


api_key = "llx-nHy5GiC5gLDi1bdQpgKrsjteSq78bVfPaDe3DDx6Sk5oVeD1"






def chunk_long_words_in_text(text, size=4000):
    """
    Splits text into chunks, each with up to 'size' words.
    """
    words = text.split()
    chunks = []
   
    for i in range(0, len(words), size):
        chunk = ' '.join(words[i:i+size])
        chunks.append(chunk)
    return chunks

sample_text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. Leading AI textbooks define the field as the study of "intelligent agents": any system that perceives its environment and takes actions that maximize its chance of achieving its goals. Some popular accounts use the term "artificial intelligence" to describe machines that mimic cognitive functions that humans associate with the human mind, such as learning and problem-solving.

As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology. Modern machine capabilities generally classified as AI include successfully understanding human speech, competing at the highest level in strategic game systems (such as chess and Go), autonomously operating cars, intelligent routing in content delivery networks, and military simulations.

Artificial intelligence was founded as an academic discipline in 1956, and in the years since has experienced several waves of optimism, followed by disappointment and the loss of funding (known as an "AI winter"), followed by new approaches, success and renewed funding. AI research has tried and discarded many different approaches since its founding, including simulating the brain, modeling human problem solving, formal logic, large databases of knowledge and imitating animal behavior. In the first decades of the 21st century, highly mathematical statistical machine learning has dominated the field, and this technique has proved highly successful, helping to solve many challenging problems throughout industry and academia.
 
The various sub-fields of AI research are centered around particular goals and the use of particular tools. The traditional goals of AI research include reasoning, knowledge representation, planning, learning, natural language processing, perception, and the ability to move and manipulate objects. General intelligence (the ability to solve an arbitrary problem) is among the field's long-term goals. To solve these problems, AI researchers have adapted and integrated a wide range of problem-solving techniques—including search and mathematical optimization, formal logic, artificial neural networks, and methods based on statistics, probability and economics. AI also draws upon computer science, psychology, linguistics, philosophy, and many other fields.
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. Leading AI textbooks define the field as the study of "intelligent agents": any system that perceives its environment and takes actions that maximize its chance of achieving its goals. Some popular accounts use the term "artificial intelligence" to describe machines that mimic cognitive functions that humans associate with the human mind, such as learning and problem-solving.

As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology. Modern machine capabilities generally classified as AI include successfully understanding human speech, competing at the highest level in strategic game systems (such as chess and Go), autonomously operating cars, intelligent routing in content delivery networks, and military simulations.

Artificial intelligence was founded as an academic discipline in 1956, and in the years since has experienced several waves of optimism, followed by disappointment and the loss of funding (known as an "AI winter"), followed by new approaches, success and renewed funding. AI research has tried and discarded many different approaches since its founding, including simulating the brain, modeling human problem solving, formal logic, large databases of knowledge and imitating animal behavior. In the first decades of the 21st century, highly mathematical statistical machine learning has dominated the field, and this technique has proved highly successful, helping to solve many challenging problems throughout industry and academia.
 
The various sub-fields of AI research are centered around particular goals and the use of particular tools. The traditional goals of AI research include reasoning, knowledge representation, planning, learning, natural language processing, perception, and the ability to move and manipulate objects. General intelligence (the ability to solve an arbitrary problem) is among the field's long-term goals. To solve these problems, AI researchers have adapted and integrated a wide range of problem-solving techniques—including search and mathematical optimization, formal logic, artificial neural networks, and methods based on statistics, probability and economics. AI also draws upon computer science, psychology, linguistics, philosophy, and many other fields.

The field was founded on the assumption that human intelligence "can be so precisely described that a machine can be made to simulate it". This raised philosophical arguments about the nature of the mind and the ethics of creating artificial beings endowed with human-like intelligence. These issues have been explored by myth, fiction and philosophy since antiquity. Computer scientists and philosophers have since suggested that AI may become an existential risk to humanity if its rational capacities are not steered towards beneficial goals.

Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. Leading AI textbooks define the field as the study of "intelligent agents": any system that perceives its environment and takes actions that maximize its chance of achieving its goals. Some popular accounts use the term "artificial intelligence" to describe machines that mimic cognitive functions that humans associate with the human mind, such as learning and problem-solving.

As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology. Modern machine capabilities generally classified as AI include successfully understanding human speech, competing at the highest level in strategic game systems (such as chess and Go), autonomously operating cars, intelligent routing in content delivery networks, and military simulations.

Artificial intelligence was founded as an academic discipline in 1956, and in the years since has experienced several waves of optimism, followed by disappointment and the loss of funding (known as an "AI winter"), followed by new approaches, success and renewed funding. AI research has tried and discarded many different approaches since its founding, including simulating the brain, modeling human problem solving, formal logic, large databases of knowledge and imitating animal behavior. In the first decades of the 21st century, highly mathematical statistical machine learning has dominated the field, and this technique has proved highly successful, helping to solve many challenging problems throughout industry and academia.
 
The various sub-fields of AI research are centered around particular goals and the use of particular tools. The traditional goals of AI research include reasoning, knowledge representation, planning, learning, natural language processing, perception, and the ability to move and manipulate objects. General intelligence (the ability to solve an arbitrary problem) is among the field's long-term goals. To solve these problems, AI researchers have adapted and integrated a wide range of problem-solving techniques—including search and mathematical optimization, formal logic, artificial neural networks, and methods based on statistics, probability and economics. AI also draws upon computer science, psychology, linguistics, philosophy, and many other fields.


"""




def chunk_word_tool(word):
    chunks = chunk_long_words_in_text(word)
    docs = [Document(page_content=str(chunk)) for chunk in chunks]
    print(docs)
    return docs





def summarize_document(final_result):
    """
    Function to summarize document and assigns task based on classification
    """
    if (final_result):
        system_prompt.format(
            input = final_result
        )

def clean_up(a):
    b = [doc.page_content for doc in a]
    for i in b:
        print(i) # Print the string content directly
        # Check if 'a' is a list
    a_is_list = isinstance(a, list)
    extracted_texts = []
    # Iterate through the list 'b' which contains the string representations of page content
    for page_str in b:
    # Use a regular expression to find the text within text='...'
        match = re.search(r"text='(.*?)'", page_str)
        if match:
    # Extract the matched text and append to the list
            extracted_texts.append(match.group(1))

    # Print the extracted text after removing newline characters
    for text in extracted_texts:
        cleaned_text = text.replace('\\n', ' ') # Replace newline characters with spaces
        return (str(cleaned_text))

def pdf_extractor():
    parser  = LlamaParse(
        api_key = api_key,
        result_type= "markdown",
        language = "en"
    )
    file_path = "C:/Users/ashle/OneDrive/Desktop/search/AshleynCastelino_Resume (3).pdf"
    result = parser.parse(file_path)

    
    final_result = chunk_word_tool(result.text)
    #text = clean_up(final_result)
    print(final_result)




a = {'position': 1, 'title': 'One Big Beautiful Bill Act 119th Congress (2025-2026)', 'link': 'https://www.congress.gov/bill/119th-congress/house-bill/1/text', 'redirect_link': 'https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.congress.gov/bill/119th-congress/house-bill/1/text&ved=2ahUKEwiZ-s3DjKiOAxURkokEHYikKhEQFnoECDQQAQ', 'displayed_link': 'https://www.congress.gov › bill › house-bill › text', 'favicon': 'https://serpapi.com/searches/686a59c74de49a71b9953c69/images/9be5cb4926096310211e25543150a5a86dd270f566e8b3d8b26996369ccd82bc.png', 'snippet': 'Text for H.R.1 - 119th Congress (2025-2026): One Big Beautiful Bill Act.', 'snippet_highlighted_words': ['Big Beautiful Bill'], 'sitelinks': {'inline': [{'title': 'Actions', 'link': 'https://www.congress.gov/bill/119th-congress/house-bill/1/all-actions'}, {'title': 'Rep. Arrington, Jodey C. [R-TX...', 'link': 'https://www.congress.gov/member/jodey-arrington/A000375'}, {'title': 'All Information', 'link': 'https://www.congress.gov/bill/119th-congress/house-bill/1/all-info'}, {'title': 'Amendments (493)', 'link': 'https://www.congress.gov/bill/119th-congress/house-bill/1/amendments'}]}, 'source': 'Congress.gov'}
link = (a.get('link'))
response = requests.get(link)
print(response.text)