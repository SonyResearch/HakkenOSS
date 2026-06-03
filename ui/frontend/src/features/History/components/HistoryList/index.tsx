/*Search history*/

import './index.css';
import { useEffect, useState } from 'react';
import { ListItem, List, ListItemButton, ListItemText } from '@mui/material';
import historyIcon from '../../../../assets/images/icons/history-icon.svg';
import { useQueryContext } from '../../../../contexts/QueryContext';
import { QueryHistoryItem } from '../../../QueryForm/components/MainForm/types';
import { useQueryService } from '../../../../services/query';
import { useQueryFormContext } from '../../../../contexts/QueryFormContext';
import { getTripleFromCondition } from '../../../QueryForm/components/MainForm/utils';
import { useConfirm } from '../../../../hooks/useConfirm';

const HistoryList = () => {
  const { getUserQueries, deleteUserQuery, clearUserQueries } =
    useQueryService();
  const [historyList, setHistoryList] = useState<QueryHistoryItem[] | []>([]);
  const { confirm, ConfirmDialog } = useConfirm();
  const { dispatch } = useQueryFormContext();
  const {
    setQuery,
    setQueryApi,
    setHypotheses,
    setVariables,
    setExample,
    setQueryMode,
    setConstraints,
  } = useQueryContext();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const getUserHistory = async () => {
      const userQueries = await getUserQueries();
      setHistoryList(userQueries);
    };
    getUserHistory();
  }, []);

  const handleSelectHistoryItem = (item: QueryHistoryItem) => {
    dispatch({ type: 'RESET' });
    setExample(null);
    setQuery(item.queryString);
    setQueryApi(JSON.stringify(item.query));
    setHypotheses(item.hypotheses);
    if (Object.values(item.hypotheses).length > 1) {
      setQueryMode('complex');
      setConstraints({});
    } else if (item.constraints) {
      setConstraints(item.constraints);
      setQueryMode('simple');
    }

    const { variable } = getTripleFromCondition(
      Object.values(item.hypotheses)[0],
    );
    const itemVariableDomain = [
      {
        label: 'X',
        domain: {
          node_domain: variable.domain,
          node_domain_id: variable.id,
        },
      },
    ];
    setVariables(itemVariableDomain);
    dispatch({
      type: 'UPDATE_VARIABLE_DOMAIN',
      value: variable.domain.replace(/ /g, '_'), //TODO: Change when we have more than one variable
    });
  };

  const handleDeleteHistoryItem = async (id: string) => {
    setError(null);
    try {
      const updatedQueries = await deleteUserQuery(id);
      setHistoryList(updatedQueries);
    } catch (error) {
      setError('Something went wrong deleting your query');
    }
  };

  const clearQueryHistory = async () => {
    const userConfirmed = await confirm([
      'Are you sure you want to delete your query history?',
      'All queries will be lost',
    ]);
    if (userConfirmed) {
      setError(null);
      try {
        await clearUserQueries();
        setHistoryList([]);
      } catch (error) {
        setError('Something went wrong deleting your history');
      }
    }
  };

  return (
    <section className="history">
      {ConfirmDialog}
      <div className="history-title">
        {' '}
        <img
          className="history-icon"
          src={historyIcon}
          alt="history icon"
        ></img>
        <u>History:</u>{' '}
        <span>
          Pull conditions from your search history to add them in your current
          query.
        </span>
      </div>

      <div className="history-list">
        <List dense sx={{ display: 'flex', flexDirection: 'column-reverse' }}>
          {!historyList || historyList.length > 0 ? (
            historyList?.map((item, index) => (
              <ListItem
                sx={{ display: 'grid', gridTemplateColumns: '10fr 1fr' }}
                key={index}
              >
                <ListItemButton onClick={() => handleSelectHistoryItem(item)}>
                  <ListItemText
                    primaryTypographyProps={{
                      sx: { fontFamily: 'arial-nova' },
                    }}
                    primary={`${index + 1} : ${item.queryString}`}
                  />
                </ListItemButton>
                <ListItemButton
                  onClick={() => handleDeleteHistoryItem(item.queryId)}
                  sx={{
                    fontSize: '0.8rem',
                    color: 'grey',
                    padding: 0,
                    justifyContent: 'flex-end',
                  }}
                >
                  delete
                </ListItemButton>
              </ListItem>
            ))
          ) : (
            <span className="empty-history">
              There are no queries in your history yet
            </span>
          )}
        </List>
      </div>
      {error && <span style={{ color: 'red' }}>{error}</span>}
      <span className="clear-history" onClick={clearQueryHistory}>
        Clear history
      </span>
    </section>
  );
};

export default HistoryList;
