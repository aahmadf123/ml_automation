import { NextRequest, NextResponse } from 'next/server';

// This is a mock API endpoint for the AI assistant
// In a real implementation, this would connect to OpenAI or another LLM service
export async function POST(request: NextRequest) {
  try {
    // Simulate API processing time
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const { message } = await request.json();
    
    if (!message) {
      return NextResponse.json(
        { error: 'Message is required' },
        { status: 400 }
      );
    }
    
    // Simple keyword-based response logic
    // In a real implementation, this would call OpenAI's API
    let response = '';
    
    if (message.toLowerCase().includes('best model')) {
      response = 'Based on the comparison metrics, the Neural Network (v1.5) is currently the best performing model with an F1 score of 0.895 and AUC of 0.95. It shows a 7.8% improvement over the baseline for F1 score and 4.4% for AUC.';
    } 
    else if (message.toLowerCase().includes('improvement') || message.toLowerCase().includes('better')) {
      response = 'The Neural Network model shows the most significant improvement over the baseline XGBoost model, with metrics improved by 7-8.5% across accuracy, precision, recall, and F1 score. The Random Forest model also shows good improvements of 3.5-4.9% across these metrics.';
    } 
    else if (message.toLowerCase().includes('fail') || message.toLowerCase().includes('error') || message.toLowerCase().includes('lightgbm')) {
      response = 'The LightGBM experimental model is currently showing as failed in the latest report. This could be due to convergence issues or parameter misconfigurations. The logs indicate possible issues with the handling of categorical features. I recommend checking the hyperparameters and ensuring proper feature encoding.';
    } 
    else if (message.toLowerCase().includes('metric') || message.toLowerCase().includes('measure')) {
      response = 'For classification models like these, F1 score is often the most balanced metric as it considers both precision and recall. If false positives are more costly, focus on precision. If false negatives are more costly, prioritize recall. For overall performance ranking, AUC provides a good threshold-independent measure of discriminative ability.';
    }
    else if (message.toLowerCase().includes('recommend') || message.toLowerCase().includes('suggest')) {
      response = 'Based on the comparison data, I recommend using the Neural Network model for production as it shows the best overall performance. However, if interpretability is important, you might consider the Random Forest model, which still shows good performance with the added benefit of being more explainable. The XGBoost baseline model remains a solid choice for systems where stability is paramount.';
    }
    else if (message.toLowerCase().includes('feature') || message.toLowerCase().includes('important')) {
      response = 'The feature importance analysis shows that "account_age" and "transaction_frequency" are the most influential features across all models. The Neural Network appears to rely more heavily on "location_risk_score" than other models do. This might explain its improved performance, as it's better utilizing the geographic risk indicators.';
    }
    else {
      response = 'I\'ve analyzed the model comparison data. The Neural Network model is outperforming others with higher precision and recall. Would you like specific metrics, feature importance details, or recommendations for improving model performance?';
    }
    
    return NextResponse.json({
      response,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('Error processing AI assistant request:', error);
    
    return NextResponse.json(
      { error: 'Failed to process request' },
      { status: 500 }
    );
  }
} 