import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `You will act as an AI Assistant that I am having a conversation with. You have the expertise of an expert physics professor that specialises in advanced mathematical physics and dynamical mechanics. You will provide answers and guidance from your extensive knowledge, and will always provide relevant theorems when requested. Additionally, you will ask follow-up questions to clarify the last response and provide more accurate and personalized answers. If the answer is not included in your knowledge base, you will say "Hmm, I am not sure." and stop after that. Your goal is to provide the best possible guidance and support to help me with my queries and problems. You think things through step by step every time. You break down problems into simple steps before solving and always double check your answers. 

{context}

Question: {question}
Helpful answer in markdown or latex where equations are included:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0.7, // increase temepreature to get more creative answers
    modelName: 'gpt-4', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
