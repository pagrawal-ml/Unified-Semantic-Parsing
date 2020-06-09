package com.ibm.sempre.test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class OvernightOriginalData {

	public static String inputFolder = "/Users/ashish/Documents/Tatzia/semantic_parsing/Overnight/overnightData";
	public static String inputFile = inputFolder + File.separator + "recipes.paraphrases.test.examples";
	
	public static String outputFolder = "/Users/ashish/Documents/Tatzia/semantic_parsing/Overnight/overnightOriginal";
	public static String outputFile = outputFolder + File.separator + "recipes_test.tsv";
	
	public static void main(String args[]) throws IOException{
		
		BufferedReader br = null;

		List<OvernightData> overnightData = new ArrayList<OvernightData>();
		br = new BufferedReader(new FileReader(inputFile));

		String line;

		boolean exampleFlag = false;
		boolean utteranceFlag = false;
		boolean originalFlag = false;
		boolean targetFormulaFlag = false;
		String utterance = null;
		String original = null;
		String targetFormula = null;
		
		while ((line = br.readLine()) != null) {
			if(line.contains("example")){
				exampleFlag = true;
			}
			
			else if(exampleFlag && line.contains("utterance")){
				String utteranceLine = line.trim().replace(")","");
				utterance = utteranceLine.substring(11);
				utteranceFlag = true;
			}else if(exampleFlag && utteranceFlag && line.contains("original")){
				String originalLine = line.trim().replace(")","");
				original = originalLine.substring(10);
				original = original.replace("\"", "");
				originalFlag = true;
			}else if(exampleFlag && utteranceFlag && originalFlag && line.contains("targetFormula")){
				targetFormulaFlag = true;
			}else if(exampleFlag && utteranceFlag && originalFlag && targetFormulaFlag){
				targetFormula  = line.trim();
				exampleFlag = false;
				utteranceFlag = false;
				originalFlag = false;
				targetFormulaFlag = false;
				
				targetFormula = targetFormula.replaceAll("edu.stanford.nlp.sempre.overnight.SimpleWorld", "SW");
				targetFormula = targetFormula.replaceAll("\\(", " ( ");
				targetFormula = targetFormula.replaceAll("\\)", " ) ");
				
				targetFormula = targetFormula.trim().replaceAll(" +", " ").trim();
				
				OvernightData data = new OvernightData();
				data.setOriginal(original);
				data.setTargetFormula(targetFormula);
				data.setUtterance(utterance);
				overnightData.add(data);
			}
		}
		br.close();
	
		
		
		FileWriter output = new FileWriter(outputFile);    
        
		for(OvernightData data: overnightData){
			output.write(data.original + "\t" + data.targetFormula + "\n");    
		}
		
		output.close(); 
		
		System.err.println("Total Queries: " + overnightData.size());
		
	}
	
}
