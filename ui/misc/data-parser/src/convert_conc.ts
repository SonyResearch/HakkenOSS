import csvParser from 'csv-parser';
import * as fs from 'fs';

interface Concept { //TODO: Delete once we don't use ocids anymore
  ocid: string;
  name: string;
  domain: string;
}
// Function to convert CSV to JSON format
function convertCsvToJson(csvFilePath: string, jsonFilePath: string): void {
  const result: Record<string, Record<string, Concept>> = {};
  let relationCount = 1; // To create keys like relation1, relation2, ...

  // Read the CSV file with tab delimiter (\t)
  fs.createReadStream(csvFilePath)
    .pipe(csvParser({ separator: '\t' })) // Set the delimiter to \t (tab)
    .on('data', (row) => {
      const relationKey = `node${relationCount}`; // Create keys like relation1, relation2, etc.
      const domainKey = (row.domain as string).toLowerCase();
      if (!(domainKey in result)) {
        result[domainKey] = {}
      }
      result[domainKey][relationKey] = {
        name: row.node, // 'name' is the relation_type field
        ocid: `id${row.ocid_node}`, // 'ocid' is the ocid_relation field
        domain: domainKey, // 'domain' is the domain field
      };
      relationCount++; // Increment for the next key
    })
    .on('end', () => {
      console.log(`Total `, Object.keys(result).length)
      // Write the result to a JSON file
      Object.entries(result).map(([domainKey, records]) => {
        fs.writeFileSync(`${jsonFilePathPrefix}-${domainKey}.ts`, `${file1stLines}${JSON.stringify(records, null, 2)}`);
        console.log('CSV successfully converted to JSON!');
      })
    });
}

// Call the function, provide the path to the CSV file and output JSON file
const csvFilePath = '/path/to/your/nodes_ui_domain.csv'; // Change this to your CSV file path
const jsonFilePathPrefix =
  '/path/to/your/output/concepts'; // Change this to your desired JSON output file path
const file1stLines = `export interface Concept {
  name: string;
  ocid: string;
  domain: string;
}

export type Concepts = Record<string, Concept>;

export const concepts: Concepts = `;
convertCsvToJson(csvFilePath, jsonFilePathPrefix);
