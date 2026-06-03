//will reuse

export interface LinkType extends d3.SimulationNodeDatum {
  relationName: string;
  source: string | NodeType;
  target: string | NodeType;
}

export interface NodeType extends d3.SimulationNodeDatum {
  id: string;
}
