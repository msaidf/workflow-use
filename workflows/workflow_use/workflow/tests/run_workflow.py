import asyncio
from pathlib import Path

from workflow_use.builder.service import BuilderService
from workflow_use.config import create_llm_pair
from workflow_use.workflow.service import Workflow

# Instantiate the LLM and the service using configuration
llm_instance, page_extraction_llm = create_llm_pair()
builder_service = BuilderService(llm=llm_instance)


async def test_run_workflow():
	"""
	Tests that the workflow is built correctly from a JSON file path.
	"""
	path = Path(__file__).parent / 'tmp' / 'recording.workflow.json'

	workflow = Workflow.load_from_file(path, llm=llm_instance, page_extraction_llm=page_extraction_llm)
	result = await workflow.run({'model': '12'})
	print(result)


if __name__ == '__main__':
	asyncio.run(test_run_workflow())
