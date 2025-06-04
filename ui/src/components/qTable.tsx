import { useEffect, useState } from "react";
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';

type Direction = "Up" | "Down" | "Left" | "Right";
type QTable = Array<Array<Direction>>;

export const QTable = () => {
    const [data, setData] = useState<QTable | null>(null);

    const fetchData = async () => {
        try {
            const response = await fetch('http://localhost:8080/q_table');
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            setData(await response.json());
        } catch (error) {
            console.error('There was a problem with the fetch operation:', error);
        }
    };

    useEffect(() => {
        // Fetch data from server on page load.
        fetchData();
    }, []);

    const renderDirection = (direction: Direction) => {
        switch (direction) {
            case "Up":
                return "⬆️";
            case "Down":
                return "⬇️";
            case "Left":
                return "⬅️";
            case "Right":
                return "➡️";
            default:
                return "";
        }
    };
    if (!data) {
        return (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
                <Typography variant="h6" sx={{ color: 'white' }}>
                    Loading Q-Table...
                </Typography>
            </div>
        );
    }
    return (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '16px' }}>
            {data && (
                <>
                    <Typography variant="h4" gutterBottom sx={{ color: 'white' }}>
                        Q-Table
                    </Typography>
                    <TableContainer 
                        component={Paper} 
                        sx={{
                            borderRadius: '8px', 
                            boxShadow: 3,
                            marginBottom: '20px'
                        }}
                    >
                        <Table sx={{ minWidth: 750, backgroundColor: '#424242' }} aria-label="q-table">
                            <TableBody>
                                {data.length > 0 &&
                                    (() => {
                                        const midRow = Math.floor(data.length / 2);
                                        const midCol = Math.floor(data[0].length / 2);
                                        return data.map((row, rowIndex) => (
                                            <TableRow key={rowIndex}>
                                                {row.map((direction, colIndex) => {
                                                    const isCheckpoint = rowIndex === midRow && colIndex === midCol;
                                                    return (
                                                        <TableCell 
                                                            key={colIndex} 
                                                            align="center" 
                                                            sx={{
                                                                color: 'white', 
                                                                border: '1px solid white',
                                                                backgroundColor: isCheckpoint ? '#4caf50' : 'inherit',
                                                                fontWeight: isCheckpoint ? 'bold' : 'normal',
                                                                fontSize: '20px',
                                                                padding: '16px'
                                                            }}
                                                        >
                                                            {isCheckpoint ? "🏁" : renderDirection(direction)}
                                                        </TableCell>
                                                    );
                                                })}
                                            </TableRow>
                                        ));
                                    })()
                                }
                            </TableBody>
                        </Table>
                    </TableContainer>
                </>
            )}
        </div>
    );
};