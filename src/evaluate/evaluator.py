import os
import re
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from typing import Any, List, TypedDict

from src.evaluate import helpers

class CodeEvaluationResult(TypedDict):
    parsed_response: str | None
    is_formatted: bool
    can_compile: bool
    pass_rate: float
    tests_passed: int
    tests_total: int
    tests_evaluated: int
    test_errors: List[str]  # Exception types for tests that failed

class CodeEvaluator:
    name: str = "code"
    debug: bool = True

    def __init__(
        self,
        num_workers: int | None = None,
        memory_per_worker: int = 1024,
        timeout: int = 3, # NOTE: If this is changed, need to re-filter the base dataset to use the new timeout
        max_failures: int = 1,
        debug: bool = False,
    ):
        self.num_workers = num_workers if num_workers is not None else int(os.environ.get('MAX_JOBS', 1)) # Maximum total number of jobs runnng at once
        self.memory_per_worker = memory_per_worker
        self.timeout = timeout
        self.debug = debug
        self.max_failures = max_failures

    def __call__(
        self, 
        response: str | None, 
        test_list: List[str] = [], # List of assertion statements to test the code
        setup_code: str = "", # Code to run before tests (e.g., imports)
        skip_parse: bool = True # Response is already parsed into code (usually True)
    ) -> CodeEvaluationResult | float:
        """
        Check if the generated program passes the given test cases.
        
        Args:
            program: The model-generated code (pure Python)
            func_name: Name of the expected function
            test_list: List of assert statements to test the function
            setup_code: Optional code to run before tests (e.g., imports)
            timeout: Time limit for each test case execution
        
        Returns:
            If return_detail is False:
                float between 0.0 and 1.0
            If return_detail is True:
                CodeEvaluationResult
        """
        result = CodeEvaluationResult(**{
            'parsed_response': None,
            'is_formatted': True,
            'can_compile': True,
            'pass_rate': 0.0,
            'tests_passed': 0,
            'tests_evaluated': 0,
            'tests_total': len(test_list),
            'test_errors': [],
        })

        # Parse the program and ensure format
        if not skip_parse:
            program = self.parse_response(response)
        else:
            program = response

        if program is None:
            result['is_formatted'] = False
            result['can_compile'] = False
            return result
        
        result['parsed_response'] = program

        # Combine all tests into a single subprocess execution
        # Create a test runner that counts tests passed and evaluated
        # Uses max_failures to stop after N failures (per sample)
        test_runner_code = helpers.create_test_runner_code(
            setup_code, 
            program,
            test_list, 
            self.max_failures
        )
        
        code_run_result: helpers.CodeRunResult = helpers.run_code_subprocess(
            test_runner_code,
            timeout=self.timeout,
            memory_limit=self.memory_per_worker,
            debug=self.debug,
        )

        # If compilation failed, return this
        result['can_compile'] = code_run_result.compiled
        if not code_run_result.compiled:
            result['test_errors'].append("MasterError: CompilationError")
        if code_run_result.timeout:
            result['test_errors'].append("MasterError: TimeoutError")
        if code_run_result.oom:
            result['test_errors'].append("MasterError: OOMError")
        if not code_run_result.success:
            result['test_errors'].append("MasterError: UnknownError: " + str(code_run_result.stdout.get('raw', 'No response')))

        # Extract the results from the stdout field
        result['tests_evaluated'] = code_run_result.stdout.get('tests_evaluated', 0)
        result['tests_passed'] = code_run_result.stdout.get('tests_passed', 0)
        result['pass_rate'] = (result['tests_passed'] / result['tests_total']) if result['tests_total'] > 0 else 0.0
        result['test_errors'] += code_run_result.stdout.get('test_errors', [])

        return result


    def batch_evaluate(
        self,
        calls: list[dict[str, Any]],
    ) -> list[CodeEvaluationResult]:
        if not calls:
            return []

        # Preserve order: collect futures with their original indices
        results: List[Any] = [None] * len(calls)

        with ThreadPoolExecutor(max_workers = self.num_workers) as executor:
            future_to_index  = {
                executor.submit(self.__call__, **call_kwargs) : idx for idx, call_kwargs in enumerate(calls)
            }

            pbar = tqdm(total = len(future_to_index), desc = "Evaluating responses")
            for future in as_completed(future_to_index):
                idx  = future_to_index[future]
                results[idx]  = future.result()
                pbar.update(1)

            pbar.close()

        return results


    def parse_response(self, response: str) -> str | None:
        # Extract all fenced python code blocks (or unlabeled) and join with double newlines
        blocks  =  re.findall(r"```(?:python)?\n(.*?)(?:```|$)", response, re.DOTALL | re.IGNORECASE)
        if not blocks:
            return None
        cleaned_blocks  =  [b.strip() for b in blocks if b.strip()]
        if not cleaned_blocks:
            return None
        return "\n\n".join(cleaned_blocks)


    def check_compile(self, response: str) -> bool:

        program = self.parse_response(response)
        if program is None:
            return False

        setup_results = helpers.run_code_subprocess(
            program,
            timeout=self.timeout,
            memory_limit=self.memory_per_worker,
            debug=self.debug,
        )

        return setup_results.compiled


    def extract_function(self, code_str: str, func_name: str) -> str:
        try:
            tree = ast.parse(code_str)
        except:
            return ""
            
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name == func_name:
                    return ast.unparse(node)

        return "" 


    def extract_function_parent(self, code_str: str, func_name: str) -> str | None:
        """
        Extract the parent class name of a function if it's a method.
        
        Returns:
            str | None: The parent class name if the function is a method of a class,
                       None if the function is not defined or has no parent class.
        """
        try:
            tree = ast.parse(code_str)
        except:
            return None
            
        # Use a visitor to track class context
        class FunctionParentExtractor(ast.NodeVisitor):
            def __init__(self, target_name: str):
                self.target_name = target_name
                self.class_name = None
                self.current_class = None
                
            def visit_ClassDef(self, node):
                # Save the current class name and visit children
                old_class = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = old_class
                
            def visit_FunctionDef(self, node):
                if node.name == self.target_name:
                    self.class_name = self.current_class
                self.generic_visit(node)
        
        extractor = FunctionParentExtractor(func_name)
        extractor.visit(tree)
        
        return extractor.class_name


    def parse_extract_function(self, response: str, func_name: str) -> str | None:
        code_str = self.parse_response(response)
        if code_str is None:
            return ""
        return self.extract_function(code_str, func_name)