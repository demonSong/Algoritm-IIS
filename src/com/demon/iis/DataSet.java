package com.demon.iis;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class DataSet {
	
	public static List<Instance> readDataSet(String path) throws FileNotFoundException{
		File file = new File(path);
		Scanner scanner = new Scanner(file);
		List<Instance> instances = new ArrayList<Instance>();
		
		while(scanner.hasNextLine()){
			String line = scanner.nextLine();
			List<String> tokens = Arrays.asList(line.split("\\s"));
			String s1 = tokens.get(0);
			int label = Integer.parseInt(s1.substring(s1.length()-1));
			int[] features = new int[tokens.size()-1];
			
			for (int i = 1; i < tokens.size(); i++)
            {
                String s = tokens.get(i);
                features[i - 1] = Integer.parseInt(s.substring(s.length() - 1));
            }
            Instance instance = new Instance(label, features);
            instances.add(instance);
        }
        scanner.close();
        return instances;
	}

}
