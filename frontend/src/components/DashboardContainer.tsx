//@ts-nocheck
import React, {useEffect, useState} from "react";
import {
    Box,
    Heading,
    Text,
    VStack,
    HStack,
    Spinner,
    SimpleGrid,
    Progress, ProgressRoot, ProgressLabel, ProgressValueText, ProgressTrack, ProgressRange, Table
} from "@chakra-ui/react";
import {
    LineChart,
    Line,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    Tooltip,
    CartesianGrid,
    PieChart,
    Pie,
    Cell,
} from "recharts";

const COLORS = ["#8884d8", "#82ca9d", "#ffc658", "#ff6f61"];

const MetricsDashboard = ({userId: any}) => {
    const [metrics, setMetrics] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchMetrics = async () => {
            try {
                const response = await fetch(`/user/${userId}/metrics`);
                const data = await response.json();
                setMetrics(data);
            } catch (error) {
                console.error("Error fetching metrics:", error);
            } finally {
                setLoading(false);
            }
        };

        fetchMetrics();
    }, [userId]);

    if (loading) {
        return (
            <Box display="flex" justifyContent="center" alignItems="center" h="100vh">
                <Spinner size="xl"/>
            </Box>
        );
    }

    if (!metrics) {
        return (
            <Box textAlign="center" p={10}>
                <Text fontSize="xl" color="red.500">
                    Unable to load metrics. Please try again later.
                </Text>
            </Box>
        );
    }

    const {
        daily_statistics,
        difficulty_ranking,
        weekly_trends,
        monthly_trends,
        forecast,
        progress,
        gap_analysis,
        memory_coefficient,
        total_words_reviewed,
        learned_words_count,
    } = metrics;

    const dailyStatsData = Object.entries(daily_statistics).map(([date, stats]) => ({
        date,
        //@ts-ignore
        ...stats,
    }));

    const progressData = Object.entries(progress).map(([status, count]) => ({
        name: status,
        value: count,
    }));

    const gapAnalysisData = Object.entries(gap_analysis).map(([date, gap]) => ({
        date,
        gap,
    }));

    return (
        <VStack p={10}>
            <Heading>Learning Metrics Dashboard</Heading>

            <SimpleGrid columns={[1, 2]} w="100%">
                <Box>
                    <Heading size="md">Daily Statistics</Heading>
                    <LineChart width={500} height={300} data={dailyStatsData}>
                        <CartesianGrid strokeDasharray="3 3"/>
                        <XAxis dataKey="date"/>
                        <YAxis/>
                        <Tooltip/>
                        <Line type="monotone" dataKey="learned" stroke="#82ca9d"/>
                        <Line type="monotone" dataKey="almost_learned" stroke="#ffc658"/>
                        <Line type="monotone" dataKey="not_learned" stroke="#ff6f61"/>
                        <Line type="monotone" dataKey="mastered" stroke="#8884d8"/>
                    </LineChart>
                </Box>

                <Box>
                    <Heading size="md">Progress Breakdown</Heading>
                    <PieChart width={400} height={400}>
                        <Pie
                            data={progressData}
                            dataKey="value"
                            nameKey="name"
                            cx="50%"
                            cy="50%"
                            outerRadius={150}
                            fill="#8884d8"
                            label
                        >
                            {progressData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]}/>
                            ))}
                        </Pie>
                    </PieChart>
                </Box>

                <Box>
                    <Heading size="md">Weekly Trends</Heading>
                    <BarChart width={500} height={300} data={[weekly_trends]}>
                        <CartesianGrid strokeDasharray="3 3"/>
                        <XAxis dataKey="name"/>
                        <YAxis/>
                        <Tooltip/>
                        <Bar dataKey="learned" fill="#82ca9d"/>
                        <Bar dataKey="almost_learned" fill="#ffc658"/>
                        <Bar dataKey="not_learned" fill="#ff6f61"/>
                        <Bar dataKey="mastered" fill="#8884d8"/>
                    </BarChart>
                </Box>

                <Box>
                    <Heading size="md">Forecast</Heading>
                    {/*<Text>Daily Target: {forecast.toFixed(2)} words/day</Text>*/}
                    <Text>Total Words Reviewed: {total_words_reviewed}</Text>
                    <Text>Learned Words Count: {learned_words_count}</Text>
                    {/*<p>{memory_coefficient}</p>*/}
                    <ProgressRoot defaultValue={40} maxW="sm" colorScheme="green" size="lg" mt={5}>
                        <HStack gap="5">
                            <ProgressLabel>Usage</ProgressLabel>
                            <ProgressTrack/>
                            <ProgressRange/>
                            <ProgressValueText>{memory_coefficient * 100}%</ProgressValueText>

                        </HStack>
                    </ProgressRoot>

                </Box>

                <Box>
                    <Heading size="md">Difficulty Ranking</Heading>

                    <Table.Root>
                        <Table.Header>
                            <Table.Row>
                                <Table.ColumnHeader>Word ID</Table.ColumnHeader>
                                <Table.ColumnHeader>Average Score</Table.ColumnHeader>
                            </Table.Row>
                        </Table.Header>

                        <Table.Body>
                            {(difficulty_ranking as []).map(([wordId, score]) => (
                                <Table.Row key={wordId}>
                                    <Table.Cell>{wordId}</Table.Cell>
                                    <Table.Cell>{score.toFixed(2)}</Table.Cell>
                                </Table.Row>
                            ))}
                        </Table.Body>
                    </Table.Root>
                </Box>

                <Box>
                    <Heading size="md">Gap Analysis</Heading>
                    <LineChart width={500} height={300} data={gapAnalysisData}>
                        <CartesianGrid strokeDasharray="3 3"/>
                        <XAxis dataKey="date"/>
                        <YAxis/>
                        <Tooltip/>
                        <Line type="monotone" dataKey="gap" stroke="#ff6f61"/>
                    </LineChart>
                </Box>

            </SimpleGrid>
        </VStack>
    );
};

export default MetricsDashboard;
