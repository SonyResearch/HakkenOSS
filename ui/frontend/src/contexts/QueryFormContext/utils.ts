import { Filters } from '../../static/filters';

export const updateSelectedFilters = (
  fixedFilters: Filters[],
  value: string,
  selectedFilters: Filters[],
) => {
  if (fixedFilters.includes(value as Filters)) return selectedFilters;
  else if (selectedFilters.includes(value as Filters))
    return selectedFilters.filter((prevFilter) => prevFilter !== value);
  else return [...selectedFilters, value];
};
