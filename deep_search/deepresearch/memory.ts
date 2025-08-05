interface Summarizer {
    summarize(items: string[]): Promise<string>;
}



type GraphNode = {
    id : string,
    content : string,
    neighbors : Set<String>;
};



export class MemoryManager{
    private memory: string[] = [];
    private summarizer: Summarizer;

    constructor(summarizer: Summarizer) {
        this.summarizer = summarizer;
    }

    async add(item : string) {
        this.memory.push(item);

        if (this.memory.length > 5) {
            console.log("More than 5");
        } 
    }
}

class SimpleSummarizer implements Summarizer {
    async summarize(items : string[]): Promise<string> {
        return `Summary ${items.join(",")}`;
    }
}

(async () => {
    const summarizer = new SimpleSummarizer();
    const manager = new MemoryManager(summarizer);

    for (let i = 1; i <= 7; i++) {
        await manager.add(`Memory item ${i}`);
    }
})();

