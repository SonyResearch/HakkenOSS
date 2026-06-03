import { createTheme } from '@mui/material';

export const conditionListTheme = createTheme({
  components: {
    MuiOutlinedInput: {
      styleOverrides: {
        input: {
          padding: '1rem',
          variants: 'standard',
          margin: 0,
          fontSize: '0.7rem',
        },
        notchedOutline: {
          border: 'none',
        },
        root: {
          padding: '0px',
          variants: 'standard',
          border: '1px solid lightgrey',
          maxHeight: '2.6rem',
          borderRadius: 'var(--border-radius)',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {},
      },
    },
    MuiStack: {
      styleOverrides: {
        root: {
          paddingRight: '1rem',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          backgroundColor: 'aliceblue',
          fontSize: '0.7rem',
          color: 'black',
        },
      },
    },
    MuiAutocomplete: {
      styleOverrides: {
        inputRoot: {
          padding: '0.2rem',
          paddingRight: '0px',
          maxHeight: '5.5rem',
          overflowY: 'scroll',
          scrollbarWidth: 'none',
        },
        root: {
          display: 'flex',
          flexDirection: 'row',
          padding: '0px',
          paddingRight: '0',
        },
      },
    },
  },
});
