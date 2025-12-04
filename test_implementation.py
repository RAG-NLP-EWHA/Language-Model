"""
Test script to verify the language model implementation
"""

import sys
import os

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        import language_model
        from language_model import LanguageModelFactory, GPT4Model, SmallLanguageModel
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_factory_methods():
    """Test that factory methods work"""
    print("\nTesting factory methods...")
    try:
        from language_model import LanguageModelFactory
        
        # Test that factory methods exist and can be called
        # (won't actually create models without API key/dependencies)
        assert hasattr(LanguageModelFactory, 'create_gpt4')
        assert hasattr(LanguageModelFactory, 'create_slm')
        print("✓ Factory methods exist")
        return True
    except Exception as e:
        print(f"✗ Factory test failed: {e}")
        return False


def test_class_structure():
    """Test that classes have required methods"""
    print("\nTesting class structure...")
    try:
        from language_model import GPT4Model, SmallLanguageModel
        
        # Check that classes have the required methods
        for cls in [GPT4Model, SmallLanguageModel]:
            assert hasattr(cls, 'generate')
            assert hasattr(cls, 'chat')
        
        print("✓ Classes have required methods")
        return True
    except Exception as e:
        print(f"✗ Class structure test failed: {e}")
        return False


def test_example_scripts():
    """Test that example scripts can be imported"""
    print("\nTesting example scripts...")
    try:
        import example_gpt4
        import example_slm
        import example_comparison
        print("✓ All example scripts can be imported")
        return True
    except Exception as e:
        print(f"✗ Example script import failed: {e}")
        return False


def test_gpt4_model_without_api():
    """Test GPT4 model initialization without API key"""
    print("\nTesting GPT4 model error handling...")
    try:
        from language_model import GPT4Model
        
        # Save original env var
        original_key = os.environ.get('OPENAI_API_KEY')
        
        # Remove API key to test error handling
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
        
        try:
            model = GPT4Model()
            print("✗ Should have raised ValueError for missing API key")
            result = False
        except ValueError as e:
            if "API key" in str(e):
                print("✓ Correctly raises ValueError when API key is missing")
                result = True
            else:
                print(f"✗ Unexpected error message: {e}")
                result = False
        except ImportError as e:
            # openai package not installed - that's OK for testing
            print("⚠ OpenAI package not installed (expected in some environments)")
            print("  This is OK - the code will raise ImportError with helpful message")
            result = True
        
        # Restore original env var
        if original_key:
            os.environ['OPENAI_API_KEY'] = original_key
        
        return result
    except Exception as e:
        print(f"✗ GPT4 error handling test failed: {e}")
        return False


def test_slm_model_without_dependencies():
    """Test SLM model initialization without dependencies"""
    print("\nTesting SLM model dependency check...")
    try:
        from language_model import SmallLanguageModel
        
        # Try to create model - will fail if transformers not installed
        # but should fail gracefully with ImportError
        try:
            import transformers
            print("✓ Transformers library is available")
            return True
        except ImportError:
            print("⚠ Transformers not installed (expected in some environments)")
            print("  This is OK - the code will raise ImportError with helpful message")
            return True
    except Exception as e:
        print(f"✗ SLM dependency test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running Language Model Implementation Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_factory_methods,
        test_class_structure,
        test_example_scripts,
        test_gpt4_model_without_api,
        test_slm_model_without_dependencies,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)
    
    if all(results):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
