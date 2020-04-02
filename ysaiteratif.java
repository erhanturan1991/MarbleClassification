package crossValidation;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Utils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.trees.J48;
import gui.ELMM;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;


public class  ysaiteratif {


	public static void main(String[] args) throws Exception {
		try {	
			Instances train = new Instances(new BufferedReader(new FileReader("C:\\Users\\Erhan\\eclipse-workspace\\mermer_kalitelendirme\\src\\crossValidation\\mix4sHistLBP.arff")));
			int lastIndex = train.numAttributes() - 1;				            
			train.setClassIndex(lastIndex);			
			MultilayerPerceptron mlp=new MultilayerPerceptron();
			//mlp.setMomentum(0.6);				
			//mlp.setLearningRate(0.9);
			mlp.setHiddenLayers("a");
			mlp.setNumDecimalPlaces(2);
			mlp.setTrainingTime(500);
			mlp.setDoNotCheckCapabilities(true);		     
			mlp.setDecay(false);
			//mlp.setGUI(true);	
			String basari= "";
			for(int i=0;i<7;i++){
				Double momentum=0.2+(0.1*i);
				mlp.setMomentum(momentum);
				for(int j=0;j<7;j++){
					Double learning=0.2+(0.1*j);
					mlp.setLearningRate(learning);
					mlp.buildClassifier(train);	
					mlp.getAutoBuild();
					//System.out.println("YSA Success Metrics: ");
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