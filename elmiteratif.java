package crossValidation;

/*This code block performs iterative cross-validation testing of the ELM machine using the Weka library.
 *Using 10-fold cross verification, the number of cells in the middle layer of the ELM machine is increased by 250,
 *starting from 250 respectively. The effect of the number of cells in the intermediate layer on the classification
 *success of the ELM is recorded by recording in series. Few of the data sets used in the study were shared to be 
 *descriptive.*/

import ELMM;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class elmiteratif {
	
	public static double round(double value, int places) {
		if (places < 0) throw new IllegalArgumentException();

		long factor = (long) Math.pow(10, places);
		value = value * factor;
		long tmp = Math.round(value);
		return (double) tmp / factor;
	}
	//Array average function
	public static double calculateAverage(Double dizi[]){ 
		double toplam = 0; 
		for (int i = 0; i < dizi.length; i++){ 
			toplam = toplam + dizi[i]; 					
		} 
		return (double) (toplam / dizi.length);

	} 

	public static double calculateStandartDeviation(Double dizi[], double ort) { 
		double kareToplam = 0; 
		for (int i = 0; i < dizi.length; i++){ 
			kareToplam = kareToplam + dizi[i] * dizi[i]; //Standart Deviation forumualtion
		} 
		return (double) Math.sqrt(kareToplam / dizi.length - ort*ort);
	} 
	
	public static void main(String[] args) throws Exception {
		// loads data and set class index
		Instances data = DataSource.read("path\\mix4sHistLBP.arff"); //Dataset path must input manually.
		data.setClassIndex(data.numAttributes()-1);

		//Extreme Learning Machine input options
		String crs="";
		ELMM elm=new ELMM();
		elm.setActiveFunction("sig"); //Another activation function sin, radbas, tansig
		elm.setNumDecimalPlaces(3);
		
		int seed = 1;
		int folds = 10;
		
		for (int m = 0; m <12; m++) {	
			
			elm.setNumberofHiddenNeurons(250+250*m);	
			System.out.println(elm.getActiveFunction());
			// randomize data
			Random rand = new Random(seed);
			Instances randData = new Instances(data);
			randData.randomize(rand);
			if (randData.classAttribute().isNominal())
				randData.stratify(folds);
	
			Double std[]=new Double[folds];
				
			Evaluation evalAll = new Evaluation(randData);
			for (int n = 0; n < folds; n++) {
				Evaluation eval = new Evaluation(randData);
				Instances train = randData.trainCV(folds, n);
				Instances test = randData.testCV(folds, n);
				
				elm.buildClassifier(train);
				eval.evaluateModel(elm, test);
				evalAll.evaluateModel(elm, test);	
				std[n]=(eval.correct()/(eval.correct()+eval.incorrect()))*100;
				
			}
			Double accuracy=evalAll.correct()/(evalAll.correct()+evalAll.incorrect());
			crs=crs+round((accuracy*100),2)+",";	
			
			// output evaluation
			System.out.println(elm.getActiveFunction());
			
			
		}
		System.out.println("Cross Valaidation:"+crs);
	}
	
	
}
