package crossValidation;

import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class j48iteratif {
	
	public static double round(double value, int places) {
		if (places < 0) throw new IllegalArgumentException();
		long factor = (long) Math.pow(10, places);
		value = value * factor;
		long tmp = Math.round(value);
		return (double) tmp / factor;
	}
	public static double ortalamaHesapla(Double dizi[]){ 
		double toplam = 0; 
		for (int i = 0; i < dizi.length; i++){ 
			toplam = toplam + dizi[i]; 					//Formül ile ortalama deðerini bulma iþlemi 
		} 
		return (double) (toplam / dizi.length); 	//Ortalama  alýndý

	} 

	public static double standartSapmaHesapla(Double dizi[], double ort) { 
		double kareToplam = 0; 
		for (int i = 0; i < dizi.length; i++){ 
			kareToplam = kareToplam + dizi[i] * dizi[i]; 			//Formül ile standart sapma bulme iþlemi
		} 
		return (double) Math.sqrt(kareToplam / dizi.length - ort*ort);	//Sapma deðeri alýndý

	} 
	
	
	public static void main(String[] args) throws Exception {
		// loads data and set class index
		Instances data = DataSource.read("C:\\Users\\Erhan\\eclipse-workspace\\mermer_kalitelendirme\\src\\crossValidation\\mix4sHistLBP.arff");
		//String clsIndex = Utils.getOption("c", args);
		//data.setClassIndex(Integer.parseInt(clsIndex) - 1);
		data.setClassIndex(data.numAttributes()-1);

		//ELM nin Ayarlarý
		String crs="";
		
		J48 j48tree = new J48();
	    j48tree.setNumDecimalPlaces(3);
		j48tree.setSubtreeRaising(true);
		j48tree.setMinNumObj(1);
		j48tree.setDoNotCheckCapabilities(false);
		//j48tree.setConfidenceFactor(Float.parseFloat("0.1"));
		j48tree.setUnpruned(true);
		
		int seed = 1;
		int folds = 10;
		
		for (int m = 1; m <10; m++) {	
			System.out.println("ERHAN TURAN");
			
			j48tree.setConfidenceFactor((float) (0.1*m));
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
				
				j48tree.buildClassifier(train);
				eval.evaluateModel(j48tree, test);
				evalAll.evaluateModel(j48tree, test);	
				std[n]=(eval.correct()/(eval.correct()+eval.incorrect()))*100;			
			}
			Double accuracy=evalAll.correct()/(evalAll.correct()+evalAll.incorrect());
			crs=crs+round((accuracy*100),2)+",";	
			
			// output evaluation
			System.out.println(j48tree.getConfidenceFactor());
			//System.out.println(evalAll.toSummaryString("=== " + folds + "-fold Cross-validation ===", false));
			/*System.out.println(evalAll.toMatrixString("=== Confusion matrix for fold " + "/" + folds + " ===\n"));		
			System.out.println("Sayýlarýn standart sapmasý: "+standartSapmaHesapla(std, ortalamaHesapla(std))); 
			System.out.println("Roc alaný:"+evalAll.areaUnderROC(0)+" "+evalAll.areaUnderROC(1));*/
			
		}
		System.out.println("Capraz Dogrulama:"+crs);
	}
	
	
}
