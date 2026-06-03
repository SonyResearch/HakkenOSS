/*We are currently not using them since we don't have relevant use cases for pubator data*/

import './index.css';

import {
  Box,
  Card,
  CardContent,
  CardActions,
  CardMedia,
  Typography,
  Button,
} from '@mui/material';

import Card1 from '../../assets/images/img1s.jpg';
import Card2 from '../../assets/images/img2s.jpg';
import Card3 from '../../assets/images/img3s.jpg';
import Card4 from '../../assets/images/img4s.jpg';
import Card5 from '../../assets/images/img5s.jpg';
import { useQueryContext } from '../../contexts/QueryContext';
import { useCases } from '../../static/useCases';
import { useQueryFormContext } from '../../contexts/QueryFormContext';
import { defaultConfig } from '../../configI';

interface ExampleCard {
  image: string;
  index: number;
}

const UseCases = () => {
  const { dispatch } = useQueryFormContext();
  const {
    setHypotheses,
    setConstraints,
    example,
    setExample,
    setVariables,
    queryMode,
  } = useQueryContext();

  if (!useCases[defaultConfig.dataSet]) return; //TODO: add use cases for Pubtator data

  const handleExampleClick = (index: number) => {
    setExample(index);
    dispatch({ type: 'RESET' });
    setVariables(useCases[defaultConfig.dataSet][index].variables);
    setHypotheses(useCases[defaultConfig.dataSet][index].hypotheses);
    if (queryMode === 'simple') {
      setConstraints(useCases[defaultConfig.dataSet][index].constraints ?? []);
    }
    dispatch({
      type: 'UPDATE_VARIABLE_DOMAIN',
      value:
        useCases[defaultConfig.dataSet][index].variables[0].domain.node_domain,
    });
  };

  const exampleImages = [Card1, Card2, Card3, Card4, Card5];

  const ExampleCard = ({ image, index }: ExampleCard) => {
    return (
      <Card className={`${example === index ? 'active' : ''} example-card`}>
        <CardMedia
          sx={{ height: '6rem' }}
          image={image}
          title={`example${index}`}
        />
        <CardContent sx={{ marginBottom: 'auto' }}>
          <Typography
            gutterBottom
            variant="h5"
            fontFamily={'arial-nova'}
            fontWeight={500}
            fontSize={'1.2rem'}
            component="div"
          >
            {useCases[defaultConfig.dataSet][index].title}
          </Typography>
          <Typography
            fontFamily={'arial-nova'}
            fontWeight={500}
            fontSize={'1rem'}
            variant="body2"
            sx={{ color: 'text.secondary' }}
          >
            {useCases[defaultConfig.dataSet][index].explanation}
          </Typography>
        </CardContent>
        <CardActions className="button-box">
          <Button
            sx={{ marginLeft: 'auto' }}
            size="small"
            onClick={() => handleExampleClick(index)}
          >
            Use Query
          </Button>
        </CardActions>
      </Card>
    );
  };

  return (
    <>
      <div className="examples">
        <h2>Use Cases</h2>
        <p className="example-text">
          Here are some example queries, designed around use cases that may be
          relevant to your work. Press &apos;USE QUERY&apos; on an example that
          interests you. The query conditions will be loaded into the query form
          below.
        </p>
      </div>

      <Box className="examples-wrapper">
        {exampleImages.map((image, index) => (
          <ExampleCard key={index} image={image} index={index}></ExampleCard>
        ))}
      </Box>
    </>
  );
};

export default UseCases;
