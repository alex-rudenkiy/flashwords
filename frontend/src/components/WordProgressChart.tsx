//@ts-nocheck
import React, {useEffect, useState} from "react";
import {Line} from "react-chartjs-2";
import "chart.js/auto";
import {Chart as ChartJS, LinearScale, PointElement, Tooltip, Legend, TimeScale} from "chart.js";
import 'chartjs-adapter-moment';
import zoomPlugin from 'chartjs-plugin-zoom';

ChartJS.register(LinearScale, PointElement, Tooltip, Legend, TimeScale, zoomPlugin);


const WordProgressChart = ({userId}: any) => {
    const [chartData, setChartData] = useState(null);

    useEffect(() => {
        fetch(`/api/stats/progress/${userId}`)
            .then((res) => res.json())
            .then((data) => {
                const datasets = Object.keys(data).map((word, index) => ({
                    label: word,
                    data: data[word].map((entry) => ({
                        x: new Date(entry.time),
                        y: entry.score,
                    })),
                    borderColor: `hsl(${(index * 360) / Object.keys(data).length}, 70%, 50%)`,
                    fill: false,
                }));

                //@ts-ignore
                setChartData({
                    datasets,
                });
            });
    }, [userId]);

    if (!chartData) return <div>Loading...</div>;

    return (
        <div>
            <h2>Word Progress Over Time</h2>
            <Line
                data={chartData}
                //@ts-ignore
                options={{
                    scales: {
                        x: {
                            type: "time", // Указываем, что ось X временная
                            time: {
                                unit: "day", // Единица времени (можно настроить)
                            },
                            title: {
                                display: true,
                                text: "Date",
                            },
                        },
                        y: {
                            title: {
                                display: true,
                                text: "Score",
                            },
                        },
                    },

                    plugins: {
                        zoom: {

                            limits: {
                                y: {min: 0, max: 10},
                            },


                            zoom: {
                                wheel: {
                                    enabled: true,
                                },
                                pinch: {
                                    enabled: true
                                },
                                mode: 'x',
                            }
                        }
                    }


                }}
            />
        </div>
    );
};

export default WordProgressChart;
