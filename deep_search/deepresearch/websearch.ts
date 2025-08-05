import { OpenAI } from "openai";
import axios from 'axios';
import { Agent } from '@mastra/core/agent';



interface SearchEngine {
    search(query : string) : Promise<string[]>;
}


class LocalArraySearch  implements SearchEngine {
    constructor(data: string[]) {
        this.data = data;
    }
    private data: string[];

    async search(query: string): Promise<string[]> {
        return Promise.resolve(
            this.data.filter(item => item.toLowerCase().includes(query.toLowerCase()))
        );
    }
    async add(query: string) {
        this.data.push(query);
    }
}
export const memoryagent = new Agent({
    
})

export class WebsearchAgent{
    /**
     * Instance of LocalArraySearch used to store and manage memory or search history locally.
     * This property provides methods for adding, searching, and retrieving items from the local memory array.
     */
    private add_memory : LocalArraySearch;
    
    constructor(memoryData: string[] = []) {
        this.add_memory = new LocalArraySearch(memoryData);
    }

    async search(query : string) : Promise<string[]> {
        const response = await axios.get(`https://api.duckduckgo.com/`, {
            params: { q: query, format: 'json' }
        });

        // DuckDuckGo API returns results in 'RelatedTopics'
        const results: string[] = (response.data.RelatedTopics || [])
            .map((topic: any) => {
                if (typeof topic.Text === 'string') {
                    return topic.Text;
                } else if (Array.isArray(topic.Topics)) {
                    return topic.Topics.map((t: any) => t.Text).filter(Boolean);
                }
                return null;
            })
            .flat()
            .filter(Boolean);

        return results;
    }
    async memory(query: string) {
        const result = await this.add_memory.search(query);
        return result;
    }
};


const app = new WebsearchAgent(['sup']);

console.log(app.memory('sup'));