import React, {useEffect} from "react";
import * as Setting from "../../Setting";
import {setToken} from "../../Setting";

interface ForeignShowSelfCheckingProps {

}

export const ForeignShowSelfChecking = () => {
  useEffect(() => {
    Setting.CasdoorSDK.exchangeForAccessToken().then((res) => {
        setToken(res.access_token);
        Setting.goToLink("/");
      }
    )
  }, []);

  return <div>signing...</div>;
}