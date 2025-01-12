//@ts-nocheck

import React, {useState} from "react";

import {useQuery, useMutation} from 'react-query';
import {Box, Button, Kbd, Spinner, Text} from '@chakra-ui/react';
import axios from 'axios';
import FancyText from '@carefully-coded/react-text-gradient';
import {useKeyPressEvent} from 'react-use';
import {getToken} from "../Setting";


export function Review() {
    const {
        data, refetch, isLoading, error, isRefetching, isFetching
    } = useQuery('nextWord', () => axios.get('/learning/next/0', {headers: {Authorization: `Bearer ${getToken()}`}}).then(res => res.data), {
        refetchOnWindowFocus: false
    });
    const mutation = useMutation(review => axios.post('/learning/review/0', review, {headers: {Authorization: `Bearer ${getToken()}`}}), {
        onSuccess: () => refetch()
    });


    const [isHiddenWord, setIsHiddenWord] = useState(true);

    const handleReview = (rating) => {
        mutation.mutate({
            wordId: data.id, rating, reviewedAt: new Date().toISOString()
        });
        setIsHiddenWord(true);
    };
    useKeyPressEvent('m', () => handleReview('mastered'));
    useKeyPressEvent(',', () => handleReview('learned'));
    useKeyPressEvent('.', () => handleReview('almost_learned'));
    useKeyPressEvent('/', () => handleReview('not_learned'));


    if (isLoading) return <Spinner/>;
    if (error) return <Text>No words available for review</Text>;


    return (<Box style={{
        display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', flexDirection: 'column'

    }}>

        {!isFetching ? <><Text fontSize="xl">{data['foreign_word']}</Text>

            <FancyText
                gradient={{from: '#17acff', to: '#ff68f0', type: 'linear'}}
                animate
                animateDuration={1000}
            >
                <Text
                    style={{
                        backgroundColor: isHiddenWord ? 'black' : 'unset',
                        cursor: isHiddenWord ? 'pointer' : 'unset'
                    }}
                    onClick={() => setIsHiddenWord(false)}>{data['native_word']}</Text>
            </FancyText>


            <Text>{data.description}</Text></> : <><Spinner/></>}

        <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignContent: 'center',
            justifyContent: 'center',
            alignItems: 'stretch',
            marginTop: '1em'
        }}>

            <div>
                <Button
                    onClick={() => handleReview('mastered')}
                    mt={4}
                    style={{display: 'block', width: '100%'}}> Mastered
                </Button>
                <Text style={{position: 'absolute', marginLeft: '10em', marginTop: '-2.1em'}}>
                    or Press <Kbd>m</Kbd>
                </Text>
            </div>


            <div>
                <Button
                    onClick={() => handleReview('learned')}
                    mt={4}
                    style={{display: 'block', width: '100%'}}> Learned
                </Button>
                <Text style={{position: 'absolute', marginLeft: '10em', marginTop: '-2.1em'}}>
                    or Press <Kbd>,</Kbd>
                </Text>
            </div>


            <div>
                <Button
                    onClick={() => handleReview('almost_learned')}
                    mt={4}
                    style={{display: 'block', width: '100%'}}> Almost Learned
                </Button>
                <Text style={{position: 'absolute', marginLeft: '10em', marginTop: '-2.1em'}}>
                    or Press <Kbd>.</Kbd>
                </Text>
            </div>


            <div>
                <Button
                    onClick={() => handleReview('not_learned')}
                    mt={4}
                    style={{display: 'block', width: '100%'}}> Not learned
                </Button>
                <Text style={{position: 'absolute', marginLeft: '10em', marginTop: '-2.1em'}}>
                    or Press <Kbd>/</Kbd>
                </Text>
            </div>


        </div>
    </Box>);
}
