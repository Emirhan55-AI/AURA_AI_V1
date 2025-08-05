from tests.test_model_accuracy import FashionModelAccuracyTester

tester = FashionModelAccuracyTester()
result = tester.test_model_accuracy('custom-fashion')
print(f'Final Accuracy: {result["accuracy"]:.1f}%')
