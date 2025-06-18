# Agents Workflow with MongoDB Persistence

This project implements a CrewAI-based workflow system with comprehensive MongoDB persistence for tracking every step of the workflow execution.

## Features

- **MongoDB Persistence**: Every step of the workflow is persisted to MongoDB for audit trails and debugging
- **Workflow Tracking**: Complete visibility into the execution flow with timestamps and step data
- **Error Handling**: Comprehensive error tracking and recovery mechanisms
- **Chat History**: Redis-based chat history management
- **RAG Integration**: Retrieval-Augmented Generation pipeline for data source queries

## Architecture

### Components

1. **MongoPersistence**: Handles all MongoDB operations for workflow state tracking
2. **AgenticHivemindFlow**: CrewAI flow that orchestrates the agent interactions
3. **run_hivemind_agent_activity**: Temporal activity that manages the workflow execution
4. **QueryDataSources**: Handles RAG queries with workflow ID tracking

### Workflow Steps Tracked

The system tracks the following steps in MongoDB:

1. **initialization**: Initial workflow setup with parameters
2. **chat_history_retrieval**: Redis chat history retrieval (if applicable)
3. **no_chat_history**: When no chat history is available
4. **flow_initialization**: AgenticHivemindFlow setup
5. **flow_execution_start**: Beginning of CrewAI flow execution
6. **local_model_classification**: Local transformer model classification result
7. **question_classification**: Language model question classification with reasoning
8. **rag_classification**: RAG question classification with score and reasoning
9. **history_query_classification**: History vs RAG query classification (if applicable)
10. **flow_execution_complete**: Completion of CrewAI flow
11. **answer_processing**: Processing of the final answer
12. **error_handling**: Any error handling steps
13. **memory_update**: Redis memory updates (if applicable)
14. **error_occurred**: Any errors during execution

## Environment Variables

Use the `.env.example` to prepare your `.env` file.

## Classification Data Persistence

The system now persists detailed classification reasoning and results for better audit trails and debugging:

### Local Model Classification
- **Step Name**: `local_model_classification`
- **Data**: Result from local transformer model
- **Model**: `local_transformer`

### Question Classification
- **Step Name**: `question_classification`
- **Data**: 
  - `result`: Boolean indicating if the message is a question
  - `reasoning`: Detailed explanation for the classification
  - `model`: `language_model`
  - `query`: Original user query

### RAG Classification
- **Step Name**: `rag_classification`
- **Data**:
  - `result`: Boolean indicating if RAG is needed
  - `score`: Sensitivity score (0-1)
  - `reasoning`: Detailed explanation for the score
  - `model`: `language_model`
  - `query`: Original user query

### History Query Classification
- **Step Name**: `history_query_classification`
- **Data**:
  - `result`: Boolean indicating if it's a history query
  - `model`: `openai_gpt4`
  - `query`: Original user query
  - `hasChatHistory`: Boolean indicating if chat history was available

## MongoDB Schema

The workflow states are stored in the `internal_messages` collection with the following structure:

```json
{
  "_id": "ObjectId",
  "communityId": "string",
  "route": {
    "source": "string",
    "destination": {
      "queue": "string",
      "event": "string"
    }
  },
  "question": {
    "message": "string",
    "filters": "object (optional)"
  },
  "response": {
    "message": "string"
  },
  "metadata": "object",
  "createdAt": "datetime",
  "updatedAt": "datetime",
  "steps": [
    {
      "stepName": "string",
      "timestamp": "datetime",
      "data": "object"
    }
  ],
  "currentStep": "string",
  "status": "string",
  "chatId": "string (optional)",
  "enableAnswerSkipping": "boolean"
}
```

## Usage

### Running the Worker

```bash
python worker.py
```

### Querying Workflow States

You can query the MongoDB collection to inspect workflow execution:

```python
from tasks.mongo_persistence import MongoPersistence

persistence = MongoPersistence()
workflow_state = persistence.get_workflow_state("workflow_id_here")
print(workflow_state)
```

## Testing

Run the unit tests:

```bash
python -m pytest tests/unit/test_mongo_persistence.py
```

## Dependencies

- `pymongo==4.8.0`: MongoDB driver
- `redis==5.2.0`: Redis client
- `crewai==0.105.0`: AI agent framework
- `temporalio`: Temporal workflow engine
- `openai==1.66.3`: OpenAI API client

## Workflow ID Tracking

The workflow ID is passed through the entire execution chain:

1. Created in `run_hivemind_agent_activity`
2. Passed to `AgenticHivemindFlow`
3. Passed to `RAGPipelineTool`
4. Passed to `QueryDataSources`
5. Included in `HivemindQueryPayload` for the `HivemindWorkflow`

This ensures complete traceability from the initial query to the final response.