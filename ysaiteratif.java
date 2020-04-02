package crossValidation;

/* In this block of code, testing was performed on the marble data set of the Artifical Neural Network
 * classifier known as MLP in the weka library. The MLP classification success of the momentum coefficient
 * was tested using 10-fold cross-validation. Testing is done by accepting the number of standard cycles as
 * 500 and the learning rate as 0.6.
 * */

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Utils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

public class  ysaiteratif {

	public static void main(String[] args) throws Exception {
		try {	
			Instances train = new Instances(new BufferedReader(new FileReader("path\\mix4sHistLBP.arff")));
			int lastIndex = train.numAttributes() - 1;				            
			train.setClassIndex(lastIndex);			
			MultilayerPerceptron mlp=new MultilayerPerceptron();
			
			mlp.setHiddenLayers("a");//Automatic neuraon count a=(attributes+classes)/2
			mlp.setNumDecimalPlaces(2);
			mlp.setTrainingTime(500);
			mlp.setDoNotCheckCapabilities(true);		     
			mlp.setDecay(false);		
			String basari= "";
			for(int i=0;i<7;i++){
				Double momentum=0.2+(0.1*i);
				mlp.setMomentum(momentum);
				for(int j=0;j<7;j++){
					Double learning=0.2+(0.1*j);
					mlp.setLearningRate(learning);
					mlp.buildClassifier(train);	
					mlp.getAutoBuild();
					//System.out.println("MLP Success Metrics: ");
					Evaluation eval = new Evaluation(train);		            
					//eval.evaluateModel(mlp, train);
					eval.crossValidateModel(mlp, train, 10, new Random(1));
					//System.out.println(eval.errorRate());
					Double accuracy=eval.correct()/(eval.correct()+eval.incorrect());
					basari=basari+accuracy*100+",";
				}	
				System.out.println(basari);
				basari="";
			}

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}				
	}

}