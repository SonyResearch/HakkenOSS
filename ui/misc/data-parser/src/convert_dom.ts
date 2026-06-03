import csvParser from 'csv-parser';
import * as fs from 'fs';

interface Domain {
  ocid: string;
  name: string;
}

// Function to convert CSV to JSON format
function convertCsvToJson(csvFilePath: string, jsonFilePath: string): void {
  const result: Record<string, Domain> = {};

  // Read the CSV file with tab delimiter (\t)
  fs.createReadStream(csvFilePath)
    .pipe(csvParser({ separator: '\t' })) // Set the delimiter to \t (tab)
    .on('data', (row) => {
      result[(row.domain as string).toLowerCase()] = {
        name: (row.domain as string).toLowerCase().replace(/_/g, ' '), // 'name' is the relation_type field
        ocid: `id${(row.ocid_domain as string).toLowerCase()}`, // 'ocid' is the ocid_relation field
      };
    })
    .on('end', () => {
      // Write the result to a JSON file
      fs.writeFileSync(jsonFilePath, JSON.stringify(result, null, 2));
      console.log('CSV successfully converted to JSON!');
    });
}

// Call the function, provide the path to the CSV file and output JSON file
const csvFilePath = '/path/to/your/domains_ui.csv'; // Change this to your CSV file path
const jsonFilePath =
  '/path/to/your/output/domains.ts'; // Change this to your desired JSON output file path

convertCsvToJson(csvFilePath, jsonFilePath);
